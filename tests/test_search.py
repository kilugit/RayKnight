import chess
import torch
from chess_ai.model import ChessNet
from chess_ai.search import MCTS

def test_mcts_search_runs():
    model = ChessNet(num_blocks=2, num_filters=64)
    model.eval()
    mcts = MCTS(model, device='cpu', c_puct=1.5)
    board = chess.Board()
    best_move, policy, wdl = mcts.search(board, num_simulations=10)
    assert best_move in board.legal_moves
    assert len(policy) == 20
    assert abs(sum(policy.values()) - 1.0) < 0.0001
    assert len(wdl) == 3
    assert abs(sum(wdl) - 1.0) < 0.0001

def test_mcts_dirichlet_noise():
    model = ChessNet(num_blocks=2, num_filters=64)
    model.eval()
    mcts = MCTS(model, device='cpu')
    board = chess.Board()
    best_move, policy, wdl = mcts.search(board, num_simulations=5, add_dirichlet_noise=True)
    assert best_move in board.legal_moves
