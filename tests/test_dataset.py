import torch
from chess_ai.dataset import ParquetChessDataset

def test_parquet_dataset_yields_batches():
    ds = ParquetChessDataset(parquet_path='dataset/dataset.parquet', batch_size=4, max_games=2, shuffle_games=False)
    batch = next(iter(ds))
    states, policies, wdl = batch
    assert states.shape == (4, 112, 8, 8)
    assert states.dtype == torch.float16
    assert policies.shape == (4,)
    assert policies.dtype == torch.long
    assert wdl.shape == (4, 3)
    assert wdl.dtype == torch.float32

def test_parquet_dataset_elo_filter():
    ds = ParquetChessDataset(parquet_path='dataset/dataset.parquet', batch_size=4, min_elo=2400, max_elo=3000, max_games=2, shuffle_games=False)
    batch = next(iter(ds))
    states, policies, wdl = batch
    assert states.shape == (4, 112, 8, 8)

def test_parquet_dataset_return_legal_mask():
    ds = ParquetChessDataset(parquet_path='dataset/dataset.parquet', batch_size=4, max_games=2, shuffle_games=False, return_legal_mask=True)
    batch = next(iter(ds))
    states, policies, wdl, masks = batch
    assert states.shape == (4, 112, 8, 8)
    assert policies.shape == (4,)
    assert wdl.shape == (4, 3)
    assert masks.shape == (4, 4096)
    assert masks.dtype == torch.bool
    for i in range(4):
        assert masks[i].sum().item() > 0
        assert masks[i, policies[i]].item() is True

def test_parquet_dataset_val_split_and_shuffle_buffer():
    ds_val = ParquetChessDataset(parquet_path='dataset/dataset.parquet', batch_size=4, max_games=2, split='val', shuffle_buffer_size=16, return_legal_mask=True)
    batch = next(iter(ds_val))
    assert len(batch) == 4
    assert batch[0].shape == (4, 112, 8, 8)

def test_parquet_dataset_single_row_group(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output_path = str(tmp_path / 'single_rg.parquet')
    records = [{'white_elo': 3500, 'black_elo': 3500, 'winner': 'white', 'moves_uci': ['e2e4', 'e7e5', 'g1f3', 'b8c6'], 'end_type': 'test', 'date': '2026.10.23', 'source': 'Stockfish'}]
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)
    ds_train = ParquetChessDataset(parquet_path=output_path, batch_size=2, split='train', return_legal_mask=True)
    batch = next(iter(ds_train))
    assert batch is not None
    assert batch[0].shape == (2, 112, 8, 8)

def test_parquet_dataset_packed_format_and_unpack_equivalence(tmp_path):
    from chess_ai.encoder import unpack_states_gpu, unpack_legal_masks_gpu
    from chess_ai.dataset_generator import save_records_to_parquet
    output_path = str(tmp_path / 'test_packed.parquet')
    records = [{'white_elo': 2800, 'black_elo': 2800, 'winner': 'draw', 'moves_uci': ['e2e4', 'e7e5', 'g1f3', 'g8f6', 'f3e5', 'd7d6'], 'end_type': 'test', 'date': '2026.10.23', 'source': 'Stockfish'}] * 5
    save_records_to_parquet(records, output_path, append=False)
    ds_dense = ParquetChessDataset(parquet_path=output_path, batch_size=4, shuffle_games=False, return_legal_mask=True, output_format='dense')
    dense_batch = next(iter(ds_dense))
    dense_states, dense_pol, dense_wdl, dense_masks = dense_batch
    ds_packed = ParquetChessDataset(parquet_path=output_path, batch_size=4, shuffle_games=False, return_legal_mask=True, output_format='packed')
    packed_batch = next(iter(ds_packed))
    pieces_bb, scalars, packed_pol, packed_wdl, masks_bb = packed_batch
    assert torch.equal(dense_pol, packed_pol)
    assert torch.allclose(dense_wdl, packed_wdl)
    unpacked_states = unpack_states_gpu(pieces_bb, scalars)
    unpacked_masks = unpack_legal_masks_gpu(masks_bb)
    assert torch.equal(dense_states, unpacked_states)
    assert torch.equal(dense_masks, unpacked_masks)
