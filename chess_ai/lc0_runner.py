import os
import subprocess
import threading
import weakref
import chess

def _cleanup_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if proc.stdin and (not proc.stdin.closed):
            try:
                proc.stdin.write('quit\n')
                proc.stdin.flush()
            except Exception:
                pass
        proc.wait(timeout=1.5)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=0.5)
        except Exception:
            pass
    finally:
        try:
            if proc.stdin and (not proc.stdin.closed):
                proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.stdout and (not proc.stdout.closed):
                proc.stdout.close()
        except Exception:
            pass

class Lc0Runner:

    def __init__(self, binary_path: str='lc0', weights_path: str | None=None, backend: str='cuda', threads: int=2, minibatch_size: int=256, max_prefetch: int | None=None, extra_args: list[str] | None=None):
        self.binary_path = binary_path
        self.weights_path = weights_path
        self.backend = backend
        self.threads = max(1, threads)
        self.minibatch_size = max(1, minibatch_size)
        self.max_prefetch = max(0, max_prefetch) if max_prefetch is not None else None
        self.extra_args = extra_args or []
        self.process: subprocess.Popen | None = None
        self._finalizer: weakref.finalize | None = None
        self.lock = threading.Lock()
        self.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.close()
        if not os.path.exists(self.binary_path):
            import shutil
            if shutil.which(self.binary_path) is None:
                raise FileNotFoundError(f'Lc0 executable not found at {self.binary_path}')
        cmd = [self.binary_path]
        if self.weights_path:
            cmd.append(f'--weights={self.weights_path}')
        if self.backend:
            cmd.append(f'--backend={self.backend}')
        if self.minibatch_size:
            cmd.append(f'--minibatch-size={self.minibatch_size}')
        if self.max_prefetch is not None and self.max_prefetch > 0:
            cmd.append(f'--max-prefetch={self.max_prefetch}')
        if self.threads:
            cmd.append(f'--threads={self.threads}')
        cmd.extend(self.extra_args)
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._finalizer = weakref.finalize(self, _cleanup_process, self.process)
        self._send_command('uci')
        self._wait_for('uciok')
        self._send_command('isready')
        self._wait_for('readyok')

    def _send_command(self, cmd: str) -> None:
        if not self.process or self.process.poll() is not None:
            err = self.process.stdout.read() if self.process and self.process.stdout else ''
            raise RuntimeError(f"Lc0 process is dead (code {(self.process.returncode if self.process else 'None')}): {err}")
        try:
            self.process.stdin.write(f'{cmd}\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            err = self.process.stdout.read() if self.process.stdout else ''
            raise RuntimeError(f'Broken pipe communicating with Lc0 (code {self.process.returncode}): {err}') from e

    def _wait_for(self, target: str) -> list[str]:
        lines = []
        if not self.process or not self.process.stdout or self.process.stdout.closed:
            raise RuntimeError(f"Lc0 process is not running while waiting for '{target}'")
        found = False
        while True:
            try:
                line = self.process.stdout.readline()
            except (ValueError, OSError):
                break
            if not line:
                if self.process.poll() is not None:
                    remaining = self.process.stdout.read() if not self.process.stdout.closed else ''
                    lines.append(remaining)
                    raise RuntimeError(f"Lc0 process exited with code {self.process.returncode} while waiting for '{target}'.\nOutput:\n" + '\n'.join(lines[-20:]))
                break
            line_str = line.strip()
            lines.append(line_str)
            if target in line_str:
                found = True
                break
        if not found:
            raise RuntimeError(f"Lc0 did not output expected '{target}'. Last output:\n" + '\n'.join(lines[-15:]))
        return lines

    def set_option(self, name: str, value: str | int | float) -> None:
        with self.lock:
            self._send_command(f'setoption name {name} value {value}')
            self._send_command('isready')
            self._wait_for('readyok')

    def set_temperature(self, temp: float, decay_moves: int=10) -> None:
        self.set_option('Temperature', temp)
        if decay_moves > 0:
            self.set_option('TempDecayMoves', decay_moves)

    def new_game(self) -> None:
        with self.lock:
            self._send_command('ucinewgame')
            self._send_command('isready')
            self._wait_for('readyok')

    def get_best_move(self, board_or_fen: chess.Board | str, nodes: int | None=100, movetime_ms: int | None=None) -> chess.Move | None:
        fen = board_or_fen.fen() if isinstance(board_or_fen, chess.Board) else board_or_fen
        with self.lock:
            self._send_command(f'position fen {fen}')
            if nodes is not None and nodes > 0:
                self._send_command(f'go nodes {nodes}')
            elif movetime_ms is not None and movetime_ms > 0:
                self._send_command(f'go movetime {movetime_ms}')
            else:
                self._send_command('go nodes 100')
            lines = self._wait_for('bestmove')
            for line in lines:
                if line.startswith('bestmove'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] != '(none)':
                        try:
                            return chess.Move.from_uci(parts[1])
                        except ValueError as e:
                            raise RuntimeError(f'Invalid UCI move received from Lc0: {parts[1]}') from e
            raise RuntimeError(f'Lc0 failed to produce a valid bestmove. Engine output:\n' + '\n'.join(lines[-10:]))

    def close(self) -> None:
        with self.lock:
            if self._finalizer:
                self._finalizer()
                self.process = None
