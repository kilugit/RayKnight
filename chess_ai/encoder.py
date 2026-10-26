import chess
import numpy as np
import torch
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_PIECE_VALUES_TUPLE = (1, 3, 3, 5, 9, 0)

def encode_piece_planes_np(board: chess.Board) -> np.ndarray:
    bbs = np.array([board.pieces_mask(1, chess.WHITE), board.pieces_mask(2, chess.WHITE), board.pieces_mask(3, chess.WHITE), board.pieces_mask(4, chess.WHITE), board.pieces_mask(5, chess.WHITE), board.pieces_mask(6, chess.WHITE), board.pieces_mask(1, chess.BLACK), board.pieces_mask(2, chess.BLACK), board.pieces_mask(3, chess.BLACK), board.pieces_mask(4, chess.BLACK), board.pieces_mask(5, chess.BLACK), board.pieces_mask(6, chess.BLACK)], dtype=np.uint64)
    return np.unpackbits(bbs.view(np.uint8), bitorder='little').astype(np.float16).reshape(12, 8, 8)

def _encode_pieces(board: chess.Board, tensor: torch.Tensor, start_plane: int) -> None:
    bits = encode_piece_planes_np(board)
    tensor[start_plane:start_plane + 12].copy_(torch.from_numpy(bits))

def encode_piece_planes(board: chess.Board) -> torch.Tensor:
    return torch.from_numpy(encode_piece_planes_np(board))

def _encode_status_planes_np(board: chess.Board, arr: np.ndarray) -> None:
    c = board.castling_rights
    if c & chess.BB_H1:
        arr[96, :, :] = 1.0
    if c & chess.BB_A1:
        arr[97, :, :] = 1.0
    if c & chess.BB_H8:
        arr[98, :, :] = 1.0
    if c & chess.BB_A8:
        arr[99, :, :] = 1.0
    if board.turn == chess.WHITE:
        arr[100, :, :] = 1.0
    if board.ep_square is not None:
        arr[101, board.ep_square // 8, board.ep_square % 8] = 1.0
    arr[102, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    arr[103, :, :] = min(board.fullmove_number / 100.0, 1.0)
    if board.is_repetition(2):
        arr[104, :, :] = 1.0
    if board.is_repetition(3):
        arr[105, :, :] = 1.0
    if board.is_check():
        arr[106, :, :] = 1.0
    w_mat = sum((board.pieces_mask(pt, chess.WHITE).bit_count() * val for pt, val in zip(range(1, 7), _PIECE_VALUES_TUPLE)))
    b_mat = sum((board.pieces_mask(pt, chess.BLACK).bit_count() * val for pt, val in zip(range(1, 7), _PIECE_VALUES_TUPLE)))
    arr[107, :, :] = w_mat / 39.0
    arr[108, :, :] = b_mat / 39.0
    if board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count() >= 2:
        arr[109, :, :] = 1.0
    if board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count() >= 2:
        arr[110, :, :] = 1.0
    arr[111, :, :] = (w_mat - b_mat) / 39.0

def _encode_status_planes(board: chess.Board, tensor: torch.Tensor) -> None:
    c = board.castling_rights
    if c & chess.BB_H1:
        tensor[96].fill_(1.0)
    if c & chess.BB_A1:
        tensor[97].fill_(1.0)
    if c & chess.BB_H8:
        tensor[98].fill_(1.0)
    if c & chess.BB_A8:
        tensor[99].fill_(1.0)
    if board.turn == chess.WHITE:
        tensor[100].fill_(1.0)
    if board.ep_square is not None:
        tensor[101, board.ep_square // 8, board.ep_square % 8] = 1.0
    tensor[102].fill_(min(board.halfmove_clock / 100.0, 1.0))
    tensor[103].fill_(min(board.fullmove_number / 100.0, 1.0))
    if board.is_repetition(2):
        tensor[104].fill_(1.0)
    if board.is_repetition(3):
        tensor[105].fill_(1.0)
    if board.is_check():
        tensor[106].fill_(1.0)
    w_mat = sum((board.pieces_mask(pt, chess.WHITE).bit_count() * val for pt, val in zip(range(1, 7), _PIECE_VALUES_TUPLE)))
    b_mat = sum((board.pieces_mask(pt, chess.BLACK).bit_count() * val for pt, val in zip(range(1, 7), _PIECE_VALUES_TUPLE)))
    tensor[107].fill_(w_mat / 39.0)
    tensor[108].fill_(b_mat / 39.0)
    if board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count() >= 2:
        tensor[109].fill_(1.0)
    if board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count() >= 2:
        tensor[110].fill_(1.0)
    tensor[111].fill_((w_mat - b_mat) / 39.0)

def encode_board_into_np(board: chess.Board, out: np.ndarray, current_planes: np.ndarray, history_planes: list[np.ndarray] | None=None) -> None:
    out.fill(0.0)
    out[:12] = current_planes
    if history_planes:
        for ply_idx, hist in enumerate(reversed(history_planes[-7:])):
            out[12 + ply_idx * 12:24 + ply_idx * 12] = hist
    _encode_status_planes_np(board, out)

def encode_board_from_planes(board: chess.Board, current_planes: torch.Tensor, history_planes: list[torch.Tensor] | None=None) -> torch.Tensor:
    tensor = torch.zeros(112, 8, 8, dtype=torch.float16)
    tensor[:12].copy_(current_planes)
    if history_planes:
        for ply_idx, hist_planes in enumerate(reversed(history_planes[-7:])):
            tensor[12 + ply_idx * 12:24 + ply_idx * 12].copy_(hist_planes)
    _encode_status_planes(board, tensor)
    return tensor

def encode_board(board: chess.Board, history: list[chess.Board] | None=None) -> torch.Tensor:
    curr_planes = encode_piece_planes(board)
    hist_planes = [encode_piece_planes(h) for h in (history[-7:] if history else [])]
    return encode_board_from_planes(board, curr_planes, hist_planes)

def get_legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = np.zeros((64, 64), dtype=bool)
    for move in board.generate_legal_moves():
        mask[move.from_square, move.to_square] = True
    return torch.from_numpy(mask)

def get_legal_move_mask_bb(board: chess.Board) -> np.ndarray:
    mask_bb = np.zeros(64, dtype=np.uint64)
    for move in board.generate_legal_moves():
        mask_bb[move.from_square] |= 1 << move.to_square
    return mask_bb.view(np.int64)

def pack_board_state(board: chess.Board, history_boards: list[chess.Board] | None=None) -> tuple[np.ndarray, np.ndarray]:
    hist = (history_boards or [])[-7:]
    all_boards = [board] + list(reversed(hist))
    pieces_bb = np.zeros((8, 12), dtype=np.uint64)
    for ply_idx, b in enumerate(all_boards):
        for i, pt in enumerate(range(1, 7)):
            pieces_bb[ply_idx, i] = b.pieces_mask(pt, chess.WHITE)
            pieces_bb[ply_idx, 6 + i] = b.pieces_mask(pt, chess.BLACK)
    c = board.castling_rights
    w_mat = sum((board.pieces_mask(pt, chess.WHITE).bit_count() * val for pt, val in zip(range(1, 7), _PIECE_VALUES_TUPLE)))
    b_mat = sum((board.pieces_mask(pt, chess.BLACK).bit_count() * val for pt, val in zip(range(1, 7), _PIECE_VALUES_TUPLE)))
    scalars = np.array([1.0 if c & chess.BB_H1 else 0.0, 1.0 if c & chess.BB_A1 else 0.0, 1.0 if c & chess.BB_H8 else 0.0, 1.0 if c & chess.BB_A8 else 0.0, 1.0 if board.turn == chess.WHITE else 0.0, float(board.ep_square // 8) if board.ep_square is not None else -1.0, float(board.ep_square % 8) if board.ep_square is not None else -1.0, min(board.halfmove_clock / 100.0, 1.0), min(board.fullmove_number / 100.0, 1.0), 1.0 if board.is_repetition(2) else 0.0, 1.0 if board.is_repetition(3) else 0.0, 1.0 if board.is_check() else 0.0, w_mat / 39.0, b_mat / 39.0, 1.0 if board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count() >= 2 else 0.0, 1.0 if board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count() >= 2 else 0.0], dtype=np.float16)
    return (pieces_bb.view(np.int64), scalars)

def unpack_states_gpu(pieces_bb: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
    device = pieces_bb.device
    B = pieces_bb.shape[0]
    shifts = torch.arange(64, dtype=torch.int64, device=device)
    states = torch.zeros(B, 112, 8, 8, dtype=torch.float16, device=device)
    states[:, :96] = (pieces_bb.unsqueeze(-1) >> shifts & 1).to(torch.float16).reshape(B, 96, 8, 8)
    for i in range(5):
        states[:, 96 + i] = scalars[:, i].view(-1, 1, 1)
    ep_row = scalars[:, 5].long()
    ep_col = scalars[:, 6].long()
    has_ep = ep_row >= 0
    if has_ep.any():
        idx = torch.where(has_ep)[0]
        states[idx, 101, ep_row[idx], ep_col[idx]] = 1.0
    for i in range(9):
        states[:, 102 + i] = scalars[:, 7 + i].view(-1, 1, 1)
    states[:, 111] = (scalars[:, 12] - scalars[:, 13]).view(-1, 1, 1)
    return states

def unpack_legal_masks_gpu(masks_bb: torch.Tensor) -> torch.Tensor:
    device = masks_bb.device
    shifts = torch.arange(64, dtype=torch.int64, device=device)
    return (masks_bb.unsqueeze(-1) >> shifts & 1 != 0).reshape(masks_bb.shape[0], 4096)

def decode_move(src_sq: int, dst_sq: int, board: chess.Board, promotion_piece: int=chess.QUEEN) -> chess.Move | None:
    move = chess.Move(src_sq, dst_sq)
    if move in board.legal_moves:
        return move
    piece = board.piece_at(src_sq)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = dst_sq // 8
        if piece.color == chess.WHITE and to_rank == 7 or (piece.color == chess.BLACK and to_rank == 0):
            prom_move = chess.Move(src_sq, dst_sq, promotion=promotion_piece)
            if prom_move in board.legal_moves:
                return prom_move
            for fallback in [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]:
                m = chess.Move(src_sq, dst_sq, promotion=fallback)
                if m in board.legal_moves:
                    return m
    return None
