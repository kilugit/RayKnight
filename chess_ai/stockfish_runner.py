import os
import shutil
import subprocess
import threading
import weakref
import chess
DEFAULT_STOCKFISH_PATH = 'stockfish/stockfish-windows-x86-64-universal.exe'

def find_stockfish_binary(custom_path: str | None=None) -> str:
    if custom_path:
        return custom_path
    env_path = os.environ.get('STOCKFISH_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists(DEFAULT_STOCKFISH_PATH):
        return DEFAULT_STOCKFISH_PATH
    fallback_win = 'stockfish/stockfish.exe'
    if os.path.exists(fallback_win):
        return fallback_win
    which_path = shutil.which('stockfish')
    if which_path:
        return which_path
    return DEFAULT_STOCKFISH_PATH

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

class StockfishRunner:

    def __init__(self, binary_path: str | None=None, threads: int=1):
        self.binary_path = find_stockfish_binary(binary_path)
        self.threads = max(1, threads)
        self.process: subprocess.Popen | None = None
        self._finalizer: weakref.finalize | None = None
        self.lock = threading.Lock()
        self.current_skill_level: int | None = None
        self.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.close()
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f'Stockfish executable not found at {self.binary_path}')
        self.process = subprocess.Popen([self.binary_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._finalizer = weakref.finalize(self, _cleanup_process, self.process)
        self._send_command('uci')
        self._wait_for('uciok')
        if self.threads > 1:
            self._send_command(f'setoption name Threads value {self.threads}')
        self._send_command('isready')
        self._wait_for('readyok')

    def _send_command(self, cmd: str) -> None:
        if self.process and self.process.stdin and (not self.process.stdin.closed):
            try:
                self.process.stdin.write(f'{cmd}\n')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def _wait_for(self, target: str) -> list[str]:
        lines = []
        if not self.process or not self.process.stdout or self.process.stdout.closed:
            return lines
        while True:
            try:
                line = self.process.stdout.readline()
            except (ValueError, OSError):
                break
            if not line:
                break
            line_str = line.strip()
            lines.append(line_str)
            if target in line_str:
                break
        return lines

    def set_option(self, name: str, value: str | int) -> None:
        with self.lock:
            self._send_command(f'setoption name {name} value {value}')
            self._send_command('isready')
            self._wait_for('readyok')

    def set_skill_level(self, level: int) -> None:
        level = max(0, min(20, level))
        if self.current_skill_level == level:
            return
        self.set_option('Skill Level', level)
        self.current_skill_level = level

    def new_game(self) -> None:
        with self.lock:
            self._send_command('ucinewgame')
            self._send_command('isready')
            self._wait_for('readyok')

    def get_best_move(self, board_or_fen: chess.Board | str, movetime_ms: int=500, depth: int | None=None) -> chess.Move | None:
        fen = board_or_fen.fen() if isinstance(board_or_fen, chess.Board) else board_or_fen
        with self.lock:
            self._send_command(f'position fen {fen}')
            if depth:
                self._send_command(f'go depth {depth}')
            else:
                self._send_command(f'go movetime {movetime_ms}')
            lines = self._wait_for('bestmove')
            for line in lines:
                if line.startswith('bestmove'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] != '(none)':
                        return chess.Move.from_uci(parts[1])
        return None

    def evaluate_position(self, board_or_fen: chess.Board | str, depth: int=10) -> dict:
        fen = board_or_fen.fen() if isinstance(board_or_fen, chess.Board) else board_or_fen
        with self.lock:
            self._send_command(f'position fen {fen}')
            self._send_command(f'go depth {depth}')
            lines = self._wait_for('bestmove')
            eval_cp = 0
            best_move = None
            for line in lines:
                if 'score cp' in line:
                    tokens = line.split()
                    idx = tokens.index('cp')
                    eval_cp = int(tokens[idx + 1])
                elif 'score mate' in line:
                    tokens = line.split()
                    idx = tokens.index('mate')
                    mate_in = int(tokens[idx + 1])
                    eval_cp = 10000 if mate_in > 0 else -10000
                if line.startswith('bestmove'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] != '(none)':
                        best_move = chess.Move.from_uci(parts[1])
        return {'eval_cp': eval_cp, 'best_move': best_move}

    def close(self) -> None:
        with self.lock:
            proc = self.process
            self.process = None
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            if proc is not None:
                _cleanup_process(proc)

    def __del__(self):
        self.close()
