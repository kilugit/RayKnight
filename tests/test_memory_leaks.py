import gc
import os
import subprocess
import tempfile
import tracemalloc
import pytest
import torch
import chess
from chess_ai.model import ChessNet
from chess_ai.search import MCTS
from chess_ai.stockfish_runner import StockfishRunner
from chess_ai.dataset_generator import save_records_to_parquet

def test_mcts_no_memory_leak():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet(num_blocks=2, num_filters=32).to(device).eval()
    if device == 'cuda':
        model.half()
    mcts = MCTS(model, device=device)
    board = chess.Board()
    mcts.search(board, num_simulations=10)
    gc.collect()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    for _ in range(20):
        mcts.search(board, num_simulations=50)
    gc.collect()
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = snap2.compare_to(snap1, 'lineno')
    total_diff = sum((stat.size_diff for stat in stats))
    assert total_diff < 5 * 1024 * 1024, f'MCTS leaked {total_diff / (1024 * 1024):.2f} MB'

def test_stockfish_runner_process_cleanup():
    sf = StockfishRunner()
    proc = sf.process
    assert proc is not None
    assert proc.poll() is None
    sf.close()
    assert sf.process is None
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail('Stockfish process did not terminate on close()')
    assert proc.poll() is not None

def test_trainer_lifecycle_memory_cleanup(tmp_path):
    from chess_ai.trainer import Trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parquet_path = str(tmp_path / 'dummy.parquet')
    records = [{'date': '2026.10.23', 'white_elo': 2500, 'black_elo': 2500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5', 'g1f3', 'b8c6'] * 5, 'end_type': 'checkmate', 'source': 'test'}] * 50
    save_records_to_parquet(records, parquet_path, append=False)
    model = ChessNet(num_blocks=2, num_filters=32).to(device)
    trainer1 = Trainer(model=model, parquet_path=parquet_path, save_dir=str(tmp_path / 'ckpts1'), batch_size=8, num_workers=0, device=device)
    trainer1.train(max_steps=5, val_every_steps=0, checkpoint_every_steps=0)
    trainer1.close()
    if device == 'cuda':
        torch.cuda.empty_cache()
        baseline_vram = torch.cuda.memory_allocated()
    trainer2 = Trainer(model=model, parquet_path=parquet_path, save_dir=str(tmp_path / 'ckpts2'), batch_size=8, num_workers=0, device=device)
    trainer2.train(max_steps=5, val_every_steps=0, checkpoint_every_steps=0)
    trainer2.close()
    if device == 'cuda':
        torch.cuda.empty_cache()
        final_vram = torch.cuda.memory_allocated()
        assert abs(final_vram - baseline_vram) < 1024 * 1024, f'VRAM grew across sessions: {baseline_vram} -> {final_vram}'
