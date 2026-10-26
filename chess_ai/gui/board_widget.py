import chess
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PyQt6.QtWidgets import QWidget, QInputDialog
UNICODE_PIECES = {chess.Piece(chess.PAWN, chess.WHITE): '♙', chess.Piece(chess.KNIGHT, chess.WHITE): '♘', chess.Piece(chess.BISHOP, chess.WHITE): '♗', chess.Piece(chess.ROOK, chess.WHITE): '♖', chess.Piece(chess.QUEEN, chess.WHITE): '♕', chess.Piece(chess.KING, chess.WHITE): '♔', chess.Piece(chess.PAWN, chess.BLACK): '♟', chess.Piece(chess.KNIGHT, chess.BLACK): '♞', chess.Piece(chess.BISHOP, chess.BLACK): '♝', chess.Piece(chess.ROOK, chess.BLACK): '♜', chess.Piece(chess.QUEEN, chess.BLACK): '♛', chess.Piece(chess.KING, chess.BLACK): '♚'}

class ChessBoardWidget(QWidget):
    move_made = pyqtSignal(chess.Move)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        self.flipped = False
        self.selected_square: int | None = None
        self.last_move: chess.Move | None = None
        self.interactive = True
        self.setMinimumSize(420, 420)
        self.light_color = QColor('#f0d9b5')
        self.dark_color = QColor('#b58863')
        self.highlight_color = QColor(205, 210, 106, 160)
        self.select_color = QColor(186, 202, 68, 200)
        self.dot_color = QColor(20, 20, 20, 80)
        self._cached_font_size: tuple[int, int] | None = None
        self._piece_font: QFont = QFont('Segoe UI Symbol', 24)
        self._coord_font: QFont = QFont('sans-serif', 9, QFont.Weight.Bold)

    def set_board(self, board: chess.Board, last_move: chess.Move | None=None):
        self.board = board
        self.last_move = last_move
        self.selected_square = None
        self.update()

    def set_flipped(self, flipped: bool):
        self.flipped = flipped
        self.update()

    def square_to_coord(self, sq: int) -> tuple[int, int]:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if self.flipped:
            col = 7 - file
            row = rank
        else:
            col = file
            row = 7 - rank
        return (col, row)

    def coord_to_square(self, col: int, row: int) -> int | None:
        if 0 <= col < 8 and 0 <= row < 8:
            if self.flipped:
                file = 7 - col
                rank = row
            else:
                file = col
                rank = 7 - row
            return chess.square(file, rank)
        return None

    def paintEvent(self, event):
        size = min(self.width(), self.height())
        if size <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        sq_size = size / 8.0
        offset_x = (self.width() - size) / 2.0
        offset_y = (self.height() - size) / 2.0
        font_size = max(1, int(sq_size * 0.75))
        coord_size = max(1, int(sq_size * 0.16))
        if self._cached_font_size != (font_size, coord_size):
            self._piece_font = QFont('Segoe UI Symbol', font_size)
            self._coord_font = QFont('sans-serif', coord_size, QFont.Weight.Bold)
            self._cached_font_size = (font_size, coord_size)
        painter.setFont(self._piece_font)
        for row in range(8):
            for col in range(8):
                sq = self.coord_to_square(col, row)
                rect = QRectF(offset_x + col * sq_size, offset_y + row * sq_size, sq_size, sq_size)
                is_light = (col + row) % 2 == 0
                bg = self.light_color if is_light else self.dark_color
                if self.last_move and sq in (self.last_move.from_square, self.last_move.to_square):
                    bg = self.highlight_color
                if sq == self.selected_square:
                    bg = self.select_color
                painter.fillRect(rect, bg)
                painter.setFont(self._coord_font)
                painter.setPen(self.dark_color if is_light else self.light_color)
                if col == 0:
                    rank_str = str(row + 1 if self.flipped else 8 - row)
                    painter.drawText(rect.adjusted(3, 2, 0, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, rank_str)
                if row == 7:
                    file_str = chr(ord('h') - col if self.flipped else ord('a') + col)
                    painter.drawText(rect.adjusted(0, 0, -3, -2), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, file_str)
                painter.setFont(self._piece_font)
                piece = self.board.piece_at(sq)
                if piece:
                    glyph = UNICODE_PIECES.get(piece, '')
                    painter.setPen(QColor('#ffffff') if piece.color == chess.WHITE else QColor('#111111'))
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, glyph)
        if self.interactive and self.selected_square is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.dot_color))
            dot_radius = sq_size * 0.16
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    col, row = self.square_to_coord(move.to_square)
                    center = QPointF(offset_x + col * sq_size + sq_size / 2.0, offset_y + row * sq_size + sq_size / 2.0)
                    painter.drawEllipse(center, dot_radius, dot_radius)

    def mousePressEvent(self, event):
        if not self.interactive or event.button() != Qt.MouseButton.LeftButton:
            return
        size = min(self.width(), self.height())
        sq_size = size / 8.0
        offset_x = (self.width() - size) / 2.0
        offset_y = (self.height() - size) / 2.0
        x = event.position().x() - offset_x
        y = event.position().y() - offset_y
        col = int(x // sq_size)
        row = int(y // sq_size)
        sq = self.coord_to_square(col, row)
        if sq is None:
            return
        if self.selected_square is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn:
                self.selected_square = sq
                self.update()
        else:
            move = chess.Move(self.selected_square, sq)
            moving_piece = self.board.piece_at(self.selected_square)
            if moving_piece and moving_piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(sq)
                if moving_piece.color == chess.WHITE and to_rank == 7 or (moving_piece.color == chess.BLACK and to_rank == 0):
                    prom_item, ok = QInputDialog.getItem(self, 'Promotion', 'Promote to:', ['Queen', 'Rook', 'Bishop', 'Knight'], 0, False)
                    prom_map = {'Queen': chess.QUEEN, 'Rook': chess.ROOK, 'Bishop': chess.BISHOP, 'Knight': chess.KNIGHT}
                    prom_type = prom_map.get(prom_item, chess.QUEEN) if ok else chess.QUEEN
                    move = chess.Move(self.selected_square, sq, promotion=prom_type)
            if move in self.board.legal_moves:
                self.selected_square = None
                self.last_move = move
                self.board.push(move)
                self.update()
                self.move_made.emit(move)
            else:
                piece = self.board.piece_at(sq)
                if piece and piece.color == self.board.turn:
                    self.selected_square = sq
                else:
                    self.selected_square = None
                self.update()
