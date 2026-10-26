import chess
import torch
from chess_ai.encoder import encode_board, get_legal_move_mask, decode_move

def test_encode_board_shape_and_dtype():
    board = chess.Board()
    t = encode_board(board)
    assert t.shape == (112, 8, 8)
    assert t.dtype == torch.float16
    assert t[100, 0, 0].item() == 1.0

def test_encode_board_history():
    board = chess.Board()
    history = []
    for move in ['e2e4', 'e7e5', 'g1f3', 'b8c6']:
        history.append(board.copy())
        board.push_san(move)
    t = encode_board(board, history)
    assert t.shape == (112, 8, 8)
    assert t[12:24].sum().item() > 0

def test_legal_move_mask():
    board = chess.Board()
    mask = get_legal_move_mask(board)
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == board.legal_moves.count() == 20

def test_decode_move_standard_and_promotion():
    board = chess.Board()
    move = decode_move(12, 28, board)
    assert move == chess.Move.from_uci('e2e4')
    prom_board = chess.Board('8/4P3/8/8/8/8/8/4K2k w - - 0 1')
    move = decode_move(chess.E7, chess.E8, prom_board)
    assert move == chess.Move.from_uci('e7e8q')

def test_encoder_bit_for_bit_equivalence():
    from chess_ai.encoder import encode_piece_planes, _encode_status_planes, PIECE_TYPES

    def reference_encode_pieces(b: chess.Board) -> torch.Tensor:
        tensor = torch.zeros(12, 8, 8, dtype=torch.float16)
        for i, piece_type in enumerate(PIECE_TYPES):
            w_bb = b.pieces_mask(piece_type, chess.WHITE)
            b_bb = b.pieces_mask(piece_type, chess.BLACK)
            for sq in chess.scan_forward(w_bb):
                tensor[i, sq // 8, sq % 8] = 1.0
            for sq in chess.scan_forward(b_bb):
                tensor[6 + i, sq // 8, sq % 8] = 1.0
        return tensor
    b = chess.Board()
    for move in ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3', 'a7a6']:
        p_fast = encode_piece_planes(b)
        p_ref = reference_encode_pieces(b)
        assert torch.equal(p_fast, p_ref), f'Mismatch in piece planes after {move}'
        t1 = torch.zeros(112, 8, 8, dtype=torch.float16)
        _encode_status_planes(b, t1)
        assert t1[100, 0, 0].item() == (1.0 if b.turn == chess.WHITE else 0.0)
        mask = get_legal_move_mask(b)
        assert mask.sum().item() == b.legal_moves.count()
        for m in b.legal_moves:
            assert mask[m.from_square, m.to_square].item() is True
        b.push_san(move)
