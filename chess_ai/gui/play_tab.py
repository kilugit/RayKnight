import chess
import chess.pgn
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QGroupBox, QLineEdit, QMessageBox, QSplitter
from chess_ai.gui.board_widget import ChessBoardWidget
from chess_ai.model import ChessNet
from chess_ai.search import MCTS
from chess_ai.stockfish_runner import StockfishRunner, find_stockfish_binary

class EngineThinker(QThread):
    finished = pyqtSignal(chess.Move, list, dict, int)

    def __init__(self, mode: str, board: chess.Board, mcts: MCTS | None, stockfish: StockfishRunner | None, think_id: int=0):
        super().__init__()
        self.mode = mode
        self.board = board.copy()
        self.mcts = mcts
        self.stockfish = stockfish
        self.think_id = think_id

    def run(self):
        if self.mode == 'Stockfish' and self.stockfish:
            best_move = self.stockfish.get_best_move(self.board, movetime_ms=500)
            eval_dict = self.stockfish.evaluate_position(self.board, depth=12)
            cp = eval_dict.get('eval_cp', 0)
            win_prob = 1.0 / (1.0 + 10 ** (-cp / 400.0))
            loss_prob = 1.0 - win_prob
            wdl = [win_prob, 0.1, loss_prob]
            self.finished.emit(best_move, wdl, {}, self.think_id)
        elif self.mcts:
            best_move, policy, wdl = self.mcts.search(self.board, num_simulations=100)
            self.finished.emit(best_move, wdl, policy, self.think_id)

class PlayTab(QWidget):

    def __init__(self, model: ChessNet | None=None, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        self.history: list[chess.Board] = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model or ChessNet(num_blocks=2, num_filters=64).to(self.device).half()
        self.model.eval()
        self.mcts = MCTS(self.model, device=self.device)
        self.stockfish_path = find_stockfish_binary()
        self.stockfish: StockfishRunner | None = None
        self._think_id = 0
        self.thinker: EngineThinker | None = None
        self._init_ui()

    def set_stockfish_path(self, path: str):
        if self.stockfish_path != path:
            self.stockfish_path = path
            if self.stockfish is not None:
                try:
                    self.stockfish.close()
                except Exception:
                    pass
                self.stockfish = None

    def _get_stockfish(self) -> StockfishRunner | None:
        if self.stockfish is not None:
            if self.stockfish_path and self.stockfish.binary_path != self.stockfish_path:
                try:
                    self.stockfish.close()
                except Exception:
                    pass
                self.stockfish = None
            elif self.stockfish.process and self.stockfish.process.poll() is not None:
                self.stockfish = None
        if self.stockfish is None:
            try:
                self.stockfish = StockfishRunner(binary_path=self.stockfish_path)
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
        self.board_widget.move_made.connect(self._on_user_move)
        left_layout.addWidget(self.board_widget, stretch=1)
        btn_row = QHBoxLayout()
        self.btn_new = QPushButton('New Game')
        self.btn_new.clicked.connect(self._new_game)
        self.btn_undo = QPushButton('Undo')
        self.btn_undo.clicked.connect(self._undo_move)
        self.btn_flip = QPushButton('Flip Board')
        self.btn_flip.clicked.connect(self._flip_board)
        btn_row.addWidget(self.btn_new)
        btn_row.addWidget(self.btn_undo)
        btn_row.addWidget(self.btn_flip)
        left_layout.addLayout(btn_row)
        splitter.addWidget(left_widget)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 0, 0, 0)
        ctrl_group = QGroupBox('Match Mode')
        ctrl_layout = QVBoxLayout(ctrl_group)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel('Opponent:'))
        self.combo_opponent = QComboBox()
        self.combo_opponent.addItems(['Ray-Knight AI', 'Stockfish', 'Human vs Human'])
        mode_row.addWidget(self.combo_opponent)
        ctrl_layout.addLayout(mode_row)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel('Human Plays:'))
        self.combo_color = QComboBox()
        self.combo_color.addItems(['White', 'Black'])
        self.combo_color.currentIndexChanged.connect(self._on_color_changed)
        color_row.addWidget(self.combo_color)
        ctrl_layout.addLayout(color_row)
        right_layout.addWidget(ctrl_group)
        eval_group = QGroupBox('Position Evaluation (W / D / L)')
        eval_layout = QVBoxLayout(eval_group)
        self.progress_win = QProgressBar()
        self.progress_win.setFormat('White Win: %p%')
        self.progress_win.setValue(33)
        self.lbl_wdl_detail = QLabel('W: 33.3% | D: 33.3% | L: 33.3%')
        self.lbl_wdl_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        eval_layout.addWidget(self.progress_win)
        eval_layout.addWidget(self.lbl_wdl_detail)
        right_layout.addWidget(eval_group)
        cand_group = QGroupBox('Top Policy Candidate Moves')
        cand_layout = QVBoxLayout(cand_group)
        self.table_candidates = QTableWidget(5, 2)
        self.table_candidates.setHorizontalHeaderLabels(['Move', 'Probability'])
        self.table_candidates.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_candidates.verticalHeader().setVisible(False)
        self.table_candidates.setMaximumHeight(140)
        cand_layout.addWidget(self.table_candidates)
        right_layout.addWidget(cand_group)
        hist_group = QGroupBox('Move History')
        hist_layout = QVBoxLayout(hist_group)
        self.table_history = QTableWidget(0, 3)
        self.table_history.setHorizontalHeaderLabels(['#', 'White', 'Black'])
        self.table_history.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_history.verticalHeader().setVisible(False)
        hist_layout.addWidget(self.table_history)
        right_layout.addWidget(hist_group)
        fen_row = QHBoxLayout()
        self.txt_fen = QLineEdit()
        self.txt_fen.setPlaceholderText('Current FEN...')
        self.txt_fen.setReadOnly(True)
        btn_copy_fen = QPushButton('Copy FEN')
        btn_copy_fen.clicked.connect(self._copy_fen)
        fen_row.addWidget(self.txt_fen)
        fen_row.addWidget(btn_copy_fen)
        right_layout.addLayout(fen_row)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)
        self._update_status()

    def _new_game(self):
        self._think_id += 1
        self.board.reset()
        self.history.clear()
        self.board_widget.set_board(self.board)
        self.table_history.setRowCount(0)
        self._update_status()
        self._check_ai_turn()

    def _undo_move(self):
        self._think_id += 1
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
        elif len(self.board.move_stack) == 1:
            self.board.pop()
        self.board_widget.set_board(self.board)
        self._refresh_history_table()
        self._update_status()

    def _flip_board(self):
        self.board_widget.set_flipped(not self.board_widget.flipped)

    def _on_color_changed(self):
        self._think_id += 1
        is_black = self.combo_color.currentText() == 'Black'
        self.board_widget.set_flipped(is_black)
        self._check_ai_turn()

    def _on_user_move(self, move: chess.Move):
        self._refresh_history_table()
        self._update_status()
        self._check_ai_turn()

    def _check_ai_turn(self):
        if self.board.is_game_over():
            return
        opponent = self.combo_opponent.currentText()
        if opponent == 'Human vs Human':
            return
        human_color = chess.WHITE if self.combo_color.currentText() == 'White' else chess.BLACK
        if self.board.turn != human_color:
            self.board_widget.interactive = False
            sf = None
            if opponent == 'Stockfish':
                sf = self._get_stockfish()
                if sf is None:
                    QMessageBox.warning(self, 'Stockfish Error', 'Stockfish executable could not be started.')
                    self.board_widget.interactive = True
                    return
            mode = 'Stockfish' if opponent == 'Stockfish' else 'Model'
            self._think_id += 1
            if self.thinker and self.thinker.isRunning():
                try:
                    self.thinker.finished.disconnect()
                except Exception:
                    pass
            self.thinker = EngineThinker(mode, self.board, self.mcts, sf, self._think_id)
            self.thinker.finished.connect(self._on_ai_move_finished)
            self.thinker.start()

    def _on_ai_move_finished(self, best_move: chess.Move | None, wdl: list, policy: dict, think_id: int=0):
        if think_id != self._think_id:
            return
        self.board_widget.interactive = True
        if best_move and best_move in self.board.legal_moves:
            self.board.push(best_move)
            self.board_widget.set_board(self.board, last_move=best_move)
            self._refresh_history_table()
        self._apply_evaluation(wdl, policy)
        self._update_status()

    def close(self):
        self._think_id += 1
        if self.thinker and self.thinker.isRunning():
            try:
                self.thinker.finished.disconnect()
            except Exception:
                pass
            self.thinker.wait(500)
            self.thinker = None
        if self.stockfish:
            self.stockfish.close()
            self.stockfish = None
        super().close()

    def _apply_evaluation(self, wdl: list, policy: dict):
        if len(wdl) == 3:
            w, d, l = wdl
            self.progress_win.setValue(int(w * 100))
            self.lbl_wdl_detail.setText(f'White: {w * 100:.1f}% | Draw: {d * 100:.1f}% | Black: {l * 100:.1f}%')
        if policy:
            sorted_moves = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:5]
            for row in range(5):
                if row < len(sorted_moves):
                    m, prob = sorted_moves[row]
                    self.table_candidates.setItem(row, 0, QTableWidgetItem(m.uci()))
                    self.table_candidates.setItem(row, 1, QTableWidgetItem(f'{prob * 100:.2f}%'))
                else:
                    self.table_candidates.setItem(row, 0, QTableWidgetItem(''))
                    self.table_candidates.setItem(row, 1, QTableWidgetItem(''))

    def _update_status(self):
        self.txt_fen.setText(self.board.fen())
        if self.board.is_checkmate():
            winner = 'Black' if self.board.turn == chess.WHITE else 'White'
            QMessageBox.information(self, 'Game Over', f'Checkmate! {winner} wins.')
        elif self.board.is_game_over():
            QMessageBox.information(self, 'Game Over', 'Game drawn.')

    def _refresh_history_table(self):
        moves = list(self.board.move_stack)
        num_turns = (len(moves) + 1) // 2
        self.table_history.setRowCount(num_turns)
        temp_board = chess.Board()
        for idx, move in enumerate(moves):
            san = temp_board.san(move)
            temp_board.push(move)
            turn_idx = idx // 2
            if idx % 2 == 0:
                self.table_history.setItem(turn_idx, 0, QTableWidgetItem(str(turn_idx + 1)))
                self.table_history.setItem(turn_idx, 1, QTableWidgetItem(san))
            else:
                self.table_history.setItem(turn_idx, 2, QTableWidgetItem(san))
        self.table_history.scrollToBottom()

    def _copy_fen(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.board.fen())
