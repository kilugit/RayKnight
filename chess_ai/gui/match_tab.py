import os
import chess
import chess.pgn
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSpinBox, QLabel, QGroupBox, QFileDialog, QMessageBox, QSplitter, QLineEdit
from chess_ai.gui.board_widget import ChessBoardWidget
from chess_ai.model import ChessNet
from chess_ai.search import MCTS
from chess_ai.stockfish_runner import StockfishRunner, find_stockfish_binary

class MatchWorker(QThread):
    move_played = pyqtSignal(chess.Board, chess.Move, str)
    game_finished = pyqtSignal(int, str, str)
    match_finished = pyqtSignal()

    def __init__(self, model: ChessNet, stockfish: StockfishRunner, total_games: int, model_time_ms: int, stockfish_time_ms: int, stockfish_skill: int, alternate_colors: bool=True):
        super().__init__()
        self.model = model
        self.stockfish = stockfish
        self.total_games = total_games
        self.model_time_ms = model_time_ms
        self.stockfish_time_ms = stockfish_time_ms
        self.stockfish_skill = stockfish_skill
        self.alternate_colors = alternate_colors
        self.running = True
        self.completed_pgns: list[chess.pgn.Game] = []

    def run(self):
        self.stockfish.set_skill_level(self.stockfish_skill)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mcts = MCTS(self.model, device=device)
        for game_idx in range(1, self.total_games + 1):
            if not self.running:
                break
            board = chess.Board()
            model_is_white = True if not self.alternate_colors else game_idx % 2 == 1
            white_name = 'RayKnight-AI' if model_is_white else f'Stockfish (Skill {self.stockfish_skill})'
            black_name = f'Stockfish (Skill {self.stockfish_skill})' if model_is_white else 'RayKnight-AI'
            pgn_game = chess.pgn.Game()
            pgn_game.headers['White'] = white_name
            pgn_game.headers['Black'] = black_name
            pgn_game.headers['Round'] = str(game_idx)
            node = pgn_game
            self.stockfish.new_game()
            history: list[chess.Board] = []
            while not board.is_game_over(claim_draw=True) and self.running:
                is_model_turn = board.turn == chess.WHITE and model_is_white or (board.turn == chess.BLACK and (not model_is_white))
                if is_model_turn:
                    sims = max(20, self.model_time_ms // 10)
                    move, _, _ = mcts.search(board, history=history, num_simulations=sims)
                    engine_name = 'RayKnight-AI'
                else:
                    move = self.stockfish.get_best_move(board, movetime_ms=self.stockfish_time_ms)
                    engine_name = 'Stockfish'
                if move is None or move not in board.legal_moves:
                    break
                history.append(board.copy())
                if len(history) > 7:
                    history.pop(0)
                board.push(move)
                node = node.add_variation(move)
                self.move_played.emit(board.copy(), move, engine_name)
                self.msleep(100)
            if not self.running:
                break
            result = board.result(claim_draw=True)
            pgn_game.headers['Result'] = result
            self.completed_pgns.append(pgn_game)
            if board.is_checkmate():
                reason = 'Checkmate'
            elif board.is_stalemate():
                reason = 'Stalemate'
            elif board.is_insufficient_material():
                reason = 'Insufficient Material'
            elif board.can_claim_threefold_repetition():
                reason = 'Threefold Repetition'
            elif board.can_claim_fifty_moves():
                reason = '50-Move Rule'
            else:
                reason = 'Draw'
            self.game_finished.emit(game_idx, result, reason)
            self.msleep(500)
        self.match_finished.emit()

    def stop(self):
        self.running = False

class MatchTab(QWidget):
    stockfish_path_changed = pyqtSignal(str)

    def __init__(self, model: ChessNet | None=None, parent=None):
        super().__init__(parent)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model or ChessNet(num_blocks=2, num_filters=64).to(self.device).half()
        self.model.eval()
        self.stockfish: StockfishRunner | None = None
        self.worker: MatchWorker | None = None
        self.model_wins = 0
        self.stockfish_wins = 0
        self.draws = 0
        self._init_ui()

    def _get_stockfish(self) -> StockfishRunner | None:
        target_path = self.edit_stockfish_path.text().strip() if hasattr(self, 'edit_stockfish_path') else None
        if self.stockfish is not None:
            if target_path and self.stockfish.binary_path != target_path:
                try:
                    self.stockfish.close()
                except Exception:
                    pass
                self.stockfish = None
            elif self.stockfish.process and self.stockfish.process.poll() is not None:
                self.stockfish = None
        if self.stockfish is None:
            try:
                self.stockfish = StockfishRunner(binary_path=target_path)
            except Exception:
                self.stockfish = None
        return self.stockfish

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.board_widget = ChessBoardWidget()
        self.board_widget.interactive = False
        left_layout.addWidget(self.board_widget, stretch=1)
        self.lbl_game_status = QLabel('Ready to start tournament match.')
        self.lbl_game_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_game_status.setStyleSheet('font-weight: bold; color: #63b3ed;')
        left_layout.addWidget(self.lbl_game_status)
        splitter.addWidget(left_widget)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 0, 0, 0)
        cfg_group = QGroupBox('Match Configuration')
        cfg_layout = QVBoxLayout(cfg_group)
        row_games = QHBoxLayout()
        row_games.addWidget(QLabel('Number of Games:'))
        self.spin_games = QSpinBox()
        self.spin_games.setRange(1, 100)
        self.spin_games.setValue(4)
        row_games.addWidget(self.spin_games)
        cfg_layout.addLayout(row_games)
        row_skill = QHBoxLayout()
        row_skill.addWidget(QLabel('Stockfish Skill Level (0-20):'))
        self.spin_skill = QSpinBox()
        self.spin_skill.setRange(0, 20)
        self.spin_skill.setValue(5)
        row_skill.addWidget(self.spin_skill)
        cfg_layout.addLayout(row_skill)
        row_time = QHBoxLayout()
        row_time.addWidget(QLabel('Time per Move (ms):'))
        self.spin_time = QSpinBox()
        self.spin_time.setRange(100, 5000)
        self.spin_time.setSingleStep(100)
        self.spin_time.setValue(300)
        row_time.addWidget(self.spin_time)
        cfg_layout.addLayout(row_time)
        row_stockfish = QHBoxLayout()
        row_stockfish.addWidget(QLabel('Stockfish Executable:'))
        self.edit_stockfish_path = QLineEdit(find_stockfish_binary())
        self.edit_stockfish_path.textChanged.connect(self._on_stockfish_text_changed)
        row_stockfish.addWidget(self.edit_stockfish_path)
        self.btn_browse_stockfish = QPushButton('Browse...')
        self.btn_browse_stockfish.clicked.connect(self._browse_stockfish_path)
        row_stockfish.addWidget(self.btn_browse_stockfish)
        cfg_layout.addLayout(row_stockfish)
        right_layout.addWidget(cfg_group)
        score_group = QGroupBox('Tournament Scoreboard')
        score_layout = QVBoxLayout(score_group)
        self.lbl_score = QLabel('Model: 0  |  Stockfish: 0  |  Draws: 0')
        self.lbl_score.setStyleSheet('font-size: 16px; font-weight: bold; color: #edf2f7;')
        self.lbl_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(self.lbl_score)
        self.lbl_winrate = QLabel('Win Rate: 0.0%')
        self.lbl_winrate.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(self.lbl_winrate)
        self.lbl_last_game = QLabel('Last Result: -')
        self.lbl_last_game.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(self.lbl_last_game)
        right_layout.addWidget(score_group)
        self.btn_start = QPushButton('Start Match')
        self.btn_start.clicked.connect(self._start_match)
        self.btn_stop = QPushButton('Stop Match')
        self.btn_stop.setObjectName('dangerButton')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_match)
        self.btn_export_pgn = QPushButton('Export Match PGN')
        self.btn_export_pgn.clicked.connect(self._export_pgn)
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)
        right_layout.addWidget(self.btn_export_pgn)
        right_layout.addStretch()
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

    def _browse_stockfish_path(self):
        curr_text = self.edit_stockfish_path.text().strip()
        start_dir = os.path.dirname(curr_text) if curr_text else '.'
        path, _ = QFileDialog.getOpenFileName(self, 'Select Stockfish Executable', start_dir, 'Executables (*.exe);;All Files (*)')
        if path:
            self.edit_stockfish_path.setText(path.replace('\\', '/'))

    def _on_stockfish_text_changed(self, text: str):
        if self.stockfish is not None:
            try:
                self.stockfish.close()
            except Exception:
                pass
            self.stockfish = None
        self.stockfish_path_changed.emit(text.strip())

    def set_stockfish_path(self, path: str):
        if self.edit_stockfish_path.text().strip() != path.strip():
            self.edit_stockfish_path.blockSignals(True)
            self.edit_stockfish_path.setText(path)
            self.edit_stockfish_path.blockSignals(False)
            if self.stockfish is not None:
                try:
                    self.stockfish.close()
                except Exception:
                    pass
                self.stockfish = None

    def _start_match(self):
        sf = self._get_stockfish()
        if not sf:
            QMessageBox.critical(self, 'Error', 'Stockfish binary not available.')
            return
        self.model_wins = 0
        self.stockfish_wins = 0
        self.draws = 0
        self._update_scoreboard()
        self.manually_stopped = False
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_game_status.setText('Match running...')
        self.worker = MatchWorker(model=self.model, stockfish=sf, total_games=self.spin_games.value(), model_time_ms=self.spin_time.value(), stockfish_time_ms=self.spin_time.value(), stockfish_skill=self.spin_skill.value())
        self.worker.move_played.connect(self._on_move_played)
        self.worker.game_finished.connect(self._on_game_finished)
        self.worker.match_finished.connect(self._on_match_finished)
        self.worker.start()

    def close(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None
        if self.stockfish:
            self.stockfish.close()
            self.stockfish = None
        super().close()

    def _stop_match(self):
        self.manually_stopped = True
        self.btn_stop.setEnabled(False)
        if self.worker:
            self.worker.stop()
        self.lbl_game_status.setText('Idle (Stopped)')

    def _on_move_played(self, board: chess.Board, move: chess.Move, engine: str):
        self.board_widget.set_board(board, last_move=move)
        ply = len(board.move_stack)
        self.lbl_game_status.setText(f'Ply {ply}: {engine} played {move.uci()}')

    def _on_game_finished(self, game_idx: int, result: str, reason: str):
        model_is_white = game_idx % 2 == 1
        if result == '1-0':
            if model_is_white:
                self.model_wins += 1
            else:
                self.stockfish_wins += 1
        elif result == '0-1':
            if model_is_white:
                self.stockfish_wins += 1
            else:
                self.model_wins += 1
        else:
            self.draws += 1
        self.lbl_last_game.setText(f'Game {game_idx}: {result} ({reason})')
        self._update_scoreboard()

    def _on_match_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if getattr(self, 'manually_stopped', False):
            self.lbl_game_status.setText('Idle')
        else:
            self.lbl_game_status.setText('Match finished!')

    def _update_scoreboard(self):
        self.lbl_score.setText(f'Model: {self.model_wins}  |  Stockfish: {self.stockfish_wins}  |  Draws: {self.draws}')
        total = self.model_wins + self.stockfish_wins + self.draws
        rate = (self.model_wins + 0.5 * self.draws) / max(1, total) * 100.0
        self.lbl_winrate.setText(f'Model Score Rate: {rate:.1f}% ({total} games)')

    def _export_pgn(self):
        if not self.worker or not self.worker.completed_pgns:
            QMessageBox.information(self, 'Export PGN', 'No completed games to export.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save Match PGN', 'match.pgn', 'PGN Files (*.pgn)')
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                for game in self.worker.completed_pgns:
                    print(game, file=f, end='\n\n')
            QMessageBox.information(self, 'Saved', f'Exported {len(self.worker.completed_pgns)} games to {path}')
