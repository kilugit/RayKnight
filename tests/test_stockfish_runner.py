import os
import chess
from chess_ai.stockfish_runner import StockfishRunner, find_stockfish_binary, DEFAULT_STOCKFISH_PATH

def test_find_stockfish_binary(monkeypatch, tmp_path):
    assert find_stockfish_binary('custom/path/stockfish.exe') == 'custom/path/stockfish.exe'
    dummy_sf = tmp_path / 'env_stockfish.exe'
    dummy_sf.touch()
    monkeypatch.setenv('STOCKFISH_PATH', str(dummy_sf))
    assert find_stockfish_binary() == str(dummy_sf)
    monkeypatch.delenv('STOCKFISH_PATH', raising=False)
    resolved = find_stockfish_binary()
    assert resolved == DEFAULT_STOCKFISH_PATH or os.path.exists(resolved)

def test_stockfish_runner_best_move():
    sf = StockfishRunner()
    board = chess.Board()
    move = sf.get_best_move(board, movetime_ms=100)
    assert move in board.legal_moves
    sf.close()

def test_stockfish_runner_tactical_mate_in_one():
    sf = StockfishRunner()
    board = chess.Board('r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4')
    move = sf.get_best_move(board, movetime_ms=200)
    assert move == chess.Move.from_uci('f3f7')
    sf.close()
