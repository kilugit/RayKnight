import os
import pyarrow.parquet as pq
import pytest
from chess_ai.dataset import ParquetChessDataset
from chess_ai.dataset_generator import StockfishDatasetGenerator, save_records_to_parquet

def test_stockfish_dataset_generator_single_game():
    generator = StockfishDatasetGenerator(movetime_ms=30, random_opening_plies=2, max_plies=8)
    from chess_ai.stockfish_runner import StockfishRunner
    sf = StockfishRunner()
    try:
        record = generator.play_game(sf)
        assert record is not None
        assert record['white_elo'] == 3500
        assert record['black_elo'] == 3500
        assert record['winner'] in ['white', 'black', 'draw']
        assert isinstance(record['moves_uci'], list)
        assert len(record['moves_uci']) > 0
        assert record['source'] == 'Stockfish Max-Strength Self-Play'
    finally:
        sf.close()

def test_stockfish_dataset_generator_saves_parquet(tmp_path):
    output_path = str(tmp_path / 'test_sf.parquet')
    generator = StockfishDatasetGenerator(movetime_ms=30, random_opening_plies=2, max_plies=8)
    completed = []

    def on_game(idx, total, rec):
        completed.append((idx, total, rec))
    total = generator.generate_games(num_games=2, output_parquet=output_path, append=False, on_game_complete=on_game)
    assert total == 2
    assert len(completed) == 2
    assert os.path.exists(output_path)
    pf = pq.ParquetFile(output_path)
    assert pf.metadata.num_rows == 2
    ds = ParquetChessDataset(parquet_path=output_path, batch_size=2, split='all', return_legal_mask=True)
    batch = next(iter(ds))
    assert batch is not None
    states, policies, wdl, masks = batch
    assert states.shape[1:] == (112, 8, 8)
    assert policies.shape[0] == 2
    assert wdl.shape[0] == 2
    assert masks.shape[1] == 4096

def test_stockfish_dataset_generator_early_stop(tmp_path):
    output_path = str(tmp_path / 'test_sf_stop.parquet')
    generator = StockfishDatasetGenerator(movetime_ms=30, random_opening_plies=2, max_plies=8)
    total = generator.generate_games(num_games=5, output_parquet=output_path, should_stop=lambda: True)
    assert total == 0

def test_stockfish_dataset_generator_randomized_skills():
    generator = StockfishDatasetGenerator(movetime_ms=30, random_opening_plies=2, max_plies=6, randomize_skills=True, min_skill=15, max_skill=20)
    from chess_ai.stockfish_runner import StockfishRunner
    sf = StockfishRunner()
    try:
        record = generator.play_game(sf)
        assert record is not None
        assert record['white_elo'] != record['black_elo']
        assert 3000 <= record['white_elo'] <= 3500
        assert 3000 <= record['black_elo'] <= 3500
        assert 'White Lvl' in record['source'] and 'Black Lvl' in record['source']
    finally:
        sf.close()

def test_stockfish_dataset_generator_parallel_workers(tmp_path):
    output_path = str(tmp_path / 'test_parallel_sf.parquet')
    generator = StockfishDatasetGenerator(movetime_ms=30, random_opening_plies=2, max_plies=6, randomize_skills=True, min_skill=15, max_skill=20)
    completed = []

    def on_game(idx, total, rec):
        completed.append((idx, total, rec))
    total = generator.generate_games(num_games=4, output_parquet=output_path, append=False, num_workers=2, on_game_complete=on_game)
    assert total == 4
    assert len(completed) == 4
    assert os.path.exists(output_path)
    pf = pq.ParquetFile(output_path)
    assert pf.metadata.num_rows == 4
    table = pf.read()
    for i in range(4):
        w_elo = table['white_elo'][i].as_py()
        b_elo = table['black_elo'][i].as_py()
        assert w_elo != b_elo
        assert 3000 <= w_elo <= 3500
        assert 3000 <= b_elo <= 3500
