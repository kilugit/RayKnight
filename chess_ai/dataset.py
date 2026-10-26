import random
import chess
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
from torch.utils.data import IterableDataset
from chess_ai.encoder import encode_piece_planes_np, encode_board_into_np, get_legal_move_mask, get_legal_move_mask_bb, pack_board_state

def _collate_batch(items: list[tuple], return_legal_mask: bool):
    if not items:
        return None
    if return_legal_mask:
        states, policies, wdls, masks = zip(*items)
        return (torch.stack(states), torch.tensor(policies, dtype=torch.long), torch.tensor(wdls, dtype=torch.float32), torch.stack(masks))
    else:
        states, policies, wdls = zip(*items)
        return (torch.stack(states), torch.tensor(policies, dtype=torch.long), torch.tensor(wdls, dtype=torch.float32))

class _SamplePool:

    def __init__(self, capacity: int, batch_size: int, packed: bool=False, return_legal_mask: bool=False):
        self.capacity = max(capacity, batch_size * 2)
        self.batch_size = batch_size
        self.packed = packed
        self.return_legal_mask = return_legal_mask
        self.free_slots: list[int] = list(range(self.capacity - 1, -1, -1))
        self.occupied_slots: list[int] = []
        if packed:
            self.pieces_bb = np.zeros((self.capacity, 8, 12), dtype=np.int64)
            self.scalars = np.zeros((self.capacity, 16), dtype=np.float16)
            self.policies = np.zeros(self.capacity, dtype=np.int64)
            self.wdls = np.zeros((self.capacity, 3), dtype=np.float32)
            if return_legal_mask:
                self.masks_bb = np.zeros((self.capacity, 64), dtype=np.int64)
        else:
            self.states = np.zeros((self.capacity, 112, 8, 8), dtype=np.float16)
            self.policies = np.zeros(self.capacity, dtype=np.int64)
            self.wdls = np.zeros((self.capacity, 3), dtype=np.float32)
            if return_legal_mask:
                self.masks = np.zeros((self.capacity, 4096), dtype=bool)

    @property
    def size(self) -> int:
        return len(self.occupied_slots)

    def add_dense(self, state_arr: np.ndarray, policy: int, wdl: list[float], mask_arr: np.ndarray | None=None):
        if not self.free_slots:
            return
        slot = self.free_slots.pop()
        self.states[slot] = state_arr
        self.policies[slot] = policy
        self.wdls[slot] = wdl
        if self.return_legal_mask and mask_arr is not None:
            self.masks[slot] = mask_arr
        self.occupied_slots.append(slot)

    def add_packed(self, pieces_bb: np.ndarray, scalars: np.ndarray, policy: int, wdl: list[float], mask_bb: np.ndarray | None=None):
        if not self.free_slots:
            return
        slot = self.free_slots.pop()
        self.pieces_bb[slot] = pieces_bb
        self.scalars[slot] = scalars
        self.policies[slot] = policy
        self.wdls[slot] = wdl
        if self.return_legal_mask and mask_bb is not None:
            self.masks_bb[slot] = mask_bb
        self.occupied_slots.append(slot)

    def pop_batch(self, count: int | None=None, shuffle: bool=True):
        total_avail = len(self.occupied_slots)
        n = count or self.batch_size
        if n > total_avail:
            n = total_avail
        if n == 0:
            return None
        if shuffle and n < total_avail:
            chosen = []
            for _ in range(n):
                idx = random.randrange(len(self.occupied_slots))
                chosen.append(self.occupied_slots[idx])
                self.occupied_slots[idx] = self.occupied_slots[-1]
                self.occupied_slots.pop()
        else:
            chosen = self.occupied_slots[:n]
            del self.occupied_slots[:n]
        self.free_slots.extend(chosen)
        if self.packed:
            batch_pieces = torch.from_numpy(self.pieces_bb[chosen].copy())
            batch_scalars = torch.from_numpy(self.scalars[chosen].copy())
            batch_pol = torch.from_numpy(self.policies[chosen].copy())
            batch_wdl = torch.from_numpy(self.wdls[chosen].copy())
            if self.return_legal_mask:
                batch_masks = torch.from_numpy(self.masks_bb[chosen].copy())
                return (batch_pieces, batch_scalars, batch_pol, batch_wdl, batch_masks)
            return (batch_pieces, batch_scalars, batch_pol, batch_wdl)
        else:
            batch_states = torch.from_numpy(self.states[chosen].copy())
            batch_pol = torch.from_numpy(self.policies[chosen].copy())
            batch_wdl = torch.from_numpy(self.wdls[chosen].copy())
            if self.return_legal_mask:
                batch_masks = torch.from_numpy(self.masks[chosen].copy())
                return (batch_states, batch_pol, batch_wdl, batch_masks)
            return (batch_states, batch_pol, batch_wdl)

class ParquetChessDataset(IterableDataset):

    def __init__(self, parquet_path: str='dataset/dataset.parquet', batch_size: int=128, min_elo: int | None=2000, max_elo: int | None=None, max_games: int | None=None, shuffle_games: bool=True, split: str='train', shuffle_buffer_size: int=0, sample_rate: float=1.0, return_legal_mask: bool=False, output_format: str='dense'):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.max_games = max_games
        self.shuffle_games = shuffle_games
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.sample_rate = sample_rate
        self.return_legal_mask = return_legal_mask
        self.output_format = output_format
        self.is_packed = output_format == 'packed'

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        all_row_groups = list(range(pf.num_row_groups))
        if len(all_row_groups) <= 1:
            row_group_indices = all_row_groups
        elif self.split == 'val':
            step = max(2, min(500, len(all_row_groups)))
            row_group_indices = [rg for rg in all_row_groups if rg % step == 0]
        elif self.split == 'train':
            step = max(2, min(500, len(all_row_groups)))
            row_group_indices = [rg for rg in all_row_groups if rg % step != 0]
            if not row_group_indices:
                row_group_indices = all_row_groups
        else:
            row_group_indices = all_row_groups
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            row_group_indices = [rg for idx, rg in enumerate(row_group_indices) if idx % worker_info.num_workers == worker_info.id]
        if (self.min_elo is not None or self.max_elo is not None) and pf.metadata is not None:
            try:
                schema_names = pf.schema_arrow.names
                w_idx = schema_names.index('white_elo') if 'white_elo' in schema_names else None
                b_idx = schema_names.index('black_elo') if 'black_elo' in schema_names else None
                if w_idx is not None and b_idx is not None:
                    pruned = []
                    for rg in row_group_indices:
                        rg_meta = pf.metadata.row_group(rg)
                        w_stat = rg_meta.column(w_idx).statistics
                        b_stat = rg_meta.column(b_idx).statistics
                        if w_stat and b_stat and w_stat.has_min_max and b_stat.has_min_max:
                            if self.min_elo is not None and (w_stat.max < self.min_elo or b_stat.max < self.min_elo):
                                continue
                            if self.max_elo is not None and (w_stat.min > self.max_elo or b_stat.min > self.max_elo):
                                continue
                        pruned.append(rg)
                    row_group_indices = pruned
            except Exception:
                pass
        if self.shuffle_games:
            random.shuffle(row_group_indices)
        games_processed = 0
        pool_capacity = max(self.shuffle_buffer_size or 0, self.batch_size * 4)
        pool = _SamplePool(capacity=pool_capacity, batch_size=self.batch_size, packed=self.is_packed, return_legal_mask=self.return_legal_mask)
        tmp_dense_state = np.zeros((112, 8, 8), dtype=np.float16) if not self.is_packed else None
        for rg_idx in row_group_indices:
            table = pf.read_row_group(rg_idx, columns=['white_elo', 'black_elo', 'winner', 'moves_uci'])
            if self.min_elo is not None or self.max_elo is not None:
                try:
                    conds = []
                    if self.min_elo is not None:
                        conds.append(pc.and_(pc.is_valid(table['white_elo']), pc.greater_equal(table['white_elo'], self.min_elo)))
                        conds.append(pc.and_(pc.is_valid(table['black_elo']), pc.greater_equal(table['black_elo'], self.min_elo)))
                    if self.max_elo is not None:
                        conds.append(pc.and_(pc.is_valid(table['white_elo']), pc.less_equal(table['white_elo'], self.max_elo)))
                        conds.append(pc.and_(pc.is_valid(table['black_elo']), pc.less_equal(table['black_elo'], self.max_elo)))
                    if conds:
                        mask = conds[0]
                        for c in conds[1:]:
                            mask = pc.and_(mask, c)
                        table = table.filter(mask)
                except Exception:
                    pass
            n_rows = len(table)
            if n_rows == 0:
                continue
            moves_col = table['moves_uci'].to_pylist()
            winner_col = table['winner'].to_pylist()
            row_indices = list(range(n_rows))
            if self.shuffle_games:
                random.shuffle(row_indices)
            for row_i in row_indices:
                if self.max_games and games_processed >= self.max_games:
                    break
                moves_uci = moves_col[row_i]
                if not moves_uci:
                    continue
                winner = winner_col[row_i]
                if winner == 'white':
                    wdl = [1.0, 0.0, 0.0]
                elif winner == 'black':
                    wdl = [0.0, 0.0, 1.0]
                else:
                    wdl = [0.0, 1.0, 0.0]
                board = chess.Board()
                history_boards: list[chess.Board] = [] if self.is_packed else None
                history_planes: list[np.ndarray] = [] if not self.is_packed else None
                for move_str in moves_uci:
                    try:
                        move = chess.Move.from_uci(move_str)
                    except ValueError:
                        break
                    target_move_idx = move.from_square << 6 | move.to_square
                    if self.return_legal_mask:
                        mask = get_legal_move_mask(board).view(-1)
                        if not mask[target_move_idx]:
                            break
                    elif move not in board.legal_moves:
                        break
                    if self.sample_rate >= 1.0 or random.random() < self.sample_rate:
                        if self.is_packed:
                            pieces_bb, scalars = pack_board_state(board, history_boards)
                            mask_bb = get_legal_move_mask_bb(board) if self.return_legal_mask else None
                            pool.add_packed(pieces_bb, scalars, target_move_idx, wdl, mask_bb)
                        else:
                            curr_planes = encode_piece_planes_np(board)
                            encode_board_into_np(board, tmp_dense_state, curr_planes, history_planes)
                            mask_arr = mask.numpy() if self.return_legal_mask else None
                            pool.add_dense(tmp_dense_state, target_move_idx, wdl, mask_arr)
                        if self.shuffle_buffer_size > 0:
                            if pool.size >= self.shuffle_buffer_size:
                                yield pool.pop_batch(shuffle=True)
                        elif pool.size >= self.batch_size:
                            yield pool.pop_batch(shuffle=False)
                    if self.is_packed:
                        history_boards.append(board.copy())
                        if len(history_boards) > 7:
                            history_boards.pop(0)
                    else:
                        if 'curr_planes' not in locals():
                            curr_planes = encode_piece_planes_np(board)
                        history_planes.append(curr_planes)
                        if len(history_planes) > 7:
                            history_planes.pop(0)
                    board.push(move)
                games_processed += 1
            if self.max_games and games_processed >= self.max_games:
                break
        while pool.size >= self.batch_size:
            yield pool.pop_batch(shuffle=self.shuffle_buffer_size > 0)
        if pool.size > 0:
            yield pool.pop_batch(count=pool.size, shuffle=False)
