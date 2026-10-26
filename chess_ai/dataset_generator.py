import concurrent.futures
import datetime
import os
import random
import threading
from typing import Callable
import chess
import pyarrow as pa
import pyarrow.parquet as pq
from chess_ai.stockfish_runner import StockfishRunner, find_stockfish_binary
PARQUET_SCHEMA = pa.schema([('date', pa.string()), ('white_elo', pa.int32()), ('black_elo', pa.int32()), ('winner', pa.string()), ('moves_uci', pa.list_(pa.string())), ('end_type', pa.string()), ('source', pa.string())])

def merge_parts_into_parquet(parts_dir: str, output_parquet: str, append: bool=True, row_group_size: int=1000) -> None:
    if not os.path.exists(parts_dir):
        return
    part_files = sorted([os.path.join(parts_dir, f) for f in os.listdir(parts_dir) if f.endswith('.parquet') and (not f.endswith('.tmp')) and (not f.endswith('.merge_tmp'))])
    if not part_files:
        try:
            os.rmdir(parts_dir)
        except OSError:
            pass
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    tmp_output = output_parquet + '.merge_tmp'
    try:
        with pq.ParquetWriter(tmp_output, schema=PARQUET_SCHEMA) as writer:
            if append and os.path.exists(output_parquet):
                with pq.ParquetFile(output_parquet) as pf:
                    for batch in pf.iter_batches(batch_size=row_group_size):
                        writer.write_batch(batch)
            for part_file in part_files:
                with pq.ParquetFile(part_file) as pf:
                    for batch in pf.iter_batches(batch_size=row_group_size):
                        writer.write_batch(batch)
        os.replace(tmp_output, output_parquet)
        for part_file in part_files:
            try:
                os.remove(part_file)
            except OSError:
                pass
        try:
            os.rmdir(parts_dir)
        except OSError:
            pass
    finally:
        if os.path.exists(tmp_output):
            try:
                os.remove(tmp_output)
            except OSError:
                pass

def save_records_to_parquet(records: list[dict], output_path: str, append: bool=True, row_group_size: int=1000) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    new_table = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)
    if append and os.path.exists(output_path):
        tmp_path = output_path + '.tmp'
        try:
            with pq.ParquetWriter(tmp_path, schema=PARQUET_SCHEMA) as writer:
                with pq.ParquetFile(output_path) as pf:
                    for batch in pf.iter_batches(batch_size=row_group_size):
                        writer.write_batch(batch)
                writer.write_table(new_table)
            os.replace(tmp_path, output_path)
            return
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            pq.write_table(new_table, output_path, row_group_size=row_group_size)
            return
    pq.write_table(new_table, output_path, row_group_size=row_group_size)

def skill_to_elo(skill: int) -> int:
    skill = max(0, min(20, skill))
    return int(1500 + skill * 100)

class StockfishDatasetGenerator:

    def __init__(self, stockfish_path: str | None=None, movetime_ms: int=100, depth: int | None=None, random_opening_plies: int=4, max_plies: int=200, threads_per_engine: int=1, randomize_skills: bool=False, min_skill: int=15, max_skill: int=20):
        self.stockfish_path = find_stockfish_binary(stockfish_path)
        self.movetime_ms = movetime_ms
        self.depth = depth
        self.random_opening_plies = max(0, random_opening_plies)
        self.max_plies = max_plies
        self.threads_per_engine = max(1, threads_per_engine)
        self.randomize_skills = randomize_skills
        self.min_skill = max(0, min(20, min_skill))
        self.max_skill = max(0, min(20, max_skill))
        if self.min_skill > self.max_skill:
            self.min_skill, self.max_skill = (self.max_skill, self.min_skill)

    def play_game(self, sf: StockfishRunner, should_stop: Callable[[], bool] | None=None, white_skill: int | None=None, black_skill: int | None=None) -> dict | None:
        if white_skill is None or black_skill is None:
            if self.randomize_skills and self.min_skill < self.max_skill:
                white_skill, black_skill = random.sample(range(self.min_skill, self.max_skill + 1), 2)
            elif self.randomize_skills:
                white_skill = self.max_skill
                black_skill = self.max_skill
            else:
                white_skill = 20
                black_skill = 20
        sf.new_game()
        board = chess.Board()
        moves_uci: list[str] = []
        for ply in range(self.max_plies):
            if should_stop and should_stop():
                return None
            if board.is_game_over(claim_draw=True):
                break
            if ply < self.random_opening_plies:
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = random.choice(legal)
            else:
                current_skill = white_skill if board.turn == chess.WHITE else black_skill
                sf.set_skill_level(current_skill)
                move = sf.get_best_move(board, movetime_ms=self.movetime_ms, depth=self.depth)
                if move is None:
                    break
            moves_uci.append(move.uci())
            board.push(move)
        if should_stop and should_stop() and (not board.is_game_over(claim_draw=True)):
            return None
        if board.is_checkmate():
            winner = 'white' if board.turn == chess.BLACK else 'black'
            end_type = 'checkmate'
        elif board.is_stalemate():
            winner = 'draw'
            end_type = 'stalemate'
        elif board.is_insufficient_material():
            winner = 'draw'
            end_type = 'insufficient_material'
        elif board.can_claim_fifty_moves() or board.is_fifty_moves():
            winner = 'draw'
            end_type = 'fifty_moves'
        elif board.can_claim_threefold_repetition() or board.is_repetition(3):
            winner = 'draw'
            end_type = 'repetition'
        elif len(moves_uci) >= self.max_plies:
            winner = 'draw'
            end_type = 'adjudication_max_plies'
        else:
            winner = 'draw'
            end_type = 'adjudication'
        if white_skill == 20 and black_skill == 20 and (not self.randomize_skills):
            source = 'Stockfish Max-Strength Self-Play'
            white_elo = 3500
            black_elo = 3500
        else:
            source = f'Stockfish Self-Play (White Lvl {white_skill} vs Black Lvl {black_skill})'
            white_elo = skill_to_elo(white_skill)
            black_elo = skill_to_elo(black_skill)
        return {'date': datetime.date.today().strftime('%Y.%m.%d'), 'white_elo': white_elo, 'black_elo': black_elo, 'winner': winner, 'moves_uci': moves_uci, 'end_type': end_type, 'source': source}

    def generate_games(self, num_games: int, output_parquet: str='dataset/stockfish_selfplay.parquet', append: bool=True, num_workers: int=1, should_stop: Callable[[], bool] | None=None, on_game_complete: Callable[[int, int, dict], None] | None=None, batch_save_every: int=5) -> int:
        parts_dir = os.path.abspath(output_parquet) + '.parts'
        if not append and os.path.exists(parts_dir):
            import shutil
            try:
                shutil.rmtree(parts_dir)
            except OSError:
                pass

        def _write_part(records: list[dict], idx: int) -> None:
            if not records:
                return
            os.makedirs(parts_dir, exist_ok=True)
            part_path = os.path.join(parts_dir, f'part_{idx:08d}.parquet')
            tbl = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)
            pq.write_table(tbl, part_path)
        if num_workers <= 1:
            sf = StockfishRunner(binary_path=self.stockfish_path, threads=self.threads_per_engine)
            pending_records: list[dict] = []
            games_generated = 0
            part_counter = 0
            try:
                for game_idx in range(1, num_games + 1):
                    if should_stop and should_stop():
                        break
                    record = self.play_game(sf, should_stop=should_stop)
                    if record is None:
                        break
                    games_generated += 1
                    pending_records.append(record)
                    if on_game_complete:
                        on_game_complete(game_idx, num_games, record)
                    if len(pending_records) >= batch_save_every:
                        part_counter += 1
                        _write_part(pending_records, part_counter)
                        pending_records.clear()
                if pending_records:
                    part_counter += 1
                    _write_part(pending_records, part_counter)
                    pending_records.clear()
            finally:
                sf.close()
                if part_counter > 0:
                    merge_parts_into_parquet(parts_dir, output_parquet, append=append)
            return games_generated
        save_lock = threading.Lock()
        game_alloc_lock = threading.Lock()
        next_game_idx = 1
        games_generated = 0
        part_counter = 0
        pending_records = []

        def worker_loop():
            nonlocal next_game_idx, games_generated, part_counter
            sf = StockfishRunner(binary_path=self.stockfish_path, threads=self.threads_per_engine)
            try:
                while True:
                    if should_stop and should_stop():
                        break
                    with game_alloc_lock:
                        if next_game_idx > num_games:
                            break
                        game_idx = next_game_idx
                        next_game_idx += 1
                    record = self.play_game(sf, should_stop=should_stop)
                    if record is None:
                        break
                    with save_lock:
                        games_generated += 1
                        completed_idx = games_generated
                        pending_records.append(record)
                        if len(pending_records) >= batch_save_every:
                            part_counter += 1
                            _write_part(pending_records, part_counter)
                            pending_records.clear()
                    if on_game_complete:
                        on_game_complete(completed_idx, num_games, record)
            finally:
                sf.close()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_loop) for _ in range(num_workers)]
                concurrent.futures.wait(futures)
            with save_lock:
                if pending_records:
                    part_counter += 1
                    _write_part(pending_records, part_counter)
                    pending_records.clear()
        finally:
            if part_counter > 0:
                merge_parts_into_parquet(parts_dir, output_parquet, append=append)
        return games_generated
        return games_generated
