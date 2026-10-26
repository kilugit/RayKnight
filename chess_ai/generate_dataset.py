import argparse
import os
import sys
import time
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from chess_ai.dataset_generator import StockfishDatasetGenerator
from chess_ai.stockfish_runner import find_stockfish_binary

def main():
    parser = argparse.ArgumentParser(description='Generate training datasets using Stockfish self-play.')
    parser.add_argument('--games', type=int, default=20, help='Number of games to generate')
    parser.add_argument('--output', type=str, default='dataset/stockfish_selfplay.parquet', help='Output parquet file path')
    parser.add_argument('--movetime', type=int, default=100, help='Stockfish move time in milliseconds')
    parser.add_argument('--depth', type=int, default=None, help='Stockfish search depth limit (optional)')
    parser.add_argument('--opening-plies', type=int, default=4, help='Number of random opening plies for game diversity')
    parser.add_argument('--max-plies', type=int, default=200, help='Maximum plies before draw adjudication')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel game workers (CPU cores)')
    parser.add_argument('--threads', type=int, default=1, help='Stockfish search threads per engine')
    parser.add_argument('--min-skill', type=int, default=15, help='Minimum Stockfish skill level for randomization (0-20)')
    parser.add_argument('--max-skill', type=int, default=20, help='Maximum Stockfish skill level for randomization (0-20)')
    parser.add_argument('--no-random-skill', action='store_true', help='Disable skill randomization (play at fixed max strength)')
    parser.add_argument('--stockfish-path', type=str, default=find_stockfish_binary(), help='Stockfish binary path')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset file instead of appending')
    args = parser.parse_args()
    randomize_skills = not args.no_random_skill
    skill_desc = f'Randomized ({args.min_skill}-{args.max_skill}, White != Black)' if randomize_skills else 'Fixed (Max Strength Level 20)'
    print(f'=== Stockfish Self-Play Dataset Generator ===')
    print(f'Games to generate:  {args.games}')
    print(f'Output path:        {args.output}')
    print(f'Move time:          {args.movetime} ms')
    print(f'Opening plies:      {args.opening_plies}')
    print(f'Parallel workers:   {args.workers} (CPU cores)')
    print(f'Threads per engine: {args.threads}')
    print(f'Skill configuration:{skill_desc}')
    print(f'Stockfish path:     {args.stockfish_path}')
    print(f"Mode:               {('Overwrite' if args.overwrite else 'Append')}")
    print('-' * 55)
    generator = StockfishDatasetGenerator(stockfish_path=args.stockfish_path, movetime_ms=args.movetime, depth=args.depth, random_opening_plies=args.opening_plies, max_plies=args.max_plies, threads_per_engine=args.threads, randomize_skills=randomize_skills, min_skill=args.min_skill, max_skill=args.max_skill)
    t0 = time.time()

    def on_game(idx, total, record):
        moves_count = len(record['moves_uci'])
        print(f"[{idx}/{total}] Outcome: {record['winner']} ({record['end_type']}, {moves_count} plies) | {record['source']}")
    total = generator.generate_games(num_games=args.games, output_parquet=args.output, append=not args.overwrite, num_workers=args.workers, on_game_complete=on_game)
    elapsed = time.time() - t0
    print('-' * 55)
    print(f'Done! Successfully generated {total} games in {elapsed:.1f}s.')
    print(f'Dataset saved to: {args.output}')
if __name__ == '__main__':
    main()
