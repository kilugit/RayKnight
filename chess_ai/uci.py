import argparse
import sys
import chess
import torch
from chess_ai.model import ChessNet, MODEL_PRESETS, find_latest_checkpoint, load_model
from chess_ai.search import MCTS

def run_uci(model_path: str | None=None, num_blocks: int | None=None, num_filters: int | None=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = model_path or find_latest_checkpoint()
    if ckpt:
        try:
            model, _ = load_model(ckpt, device=device)
        except Exception:
            cfg = {'num_blocks': num_blocks or 10, 'num_filters': num_filters or 128}
            model = ChessNet(**cfg).to(device)
    else:
        cfg = {'num_blocks': num_blocks or 10, 'num_filters': num_filters or 128}
        model = ChessNet(**cfg).to(device)
    model.eval()
    if device == 'cuda':
        model.half()
    mcts = MCTS(model, device=device)
    board = chess.Board()
    history: list[chess.Board] = []
    engine_tag = 'RayKnight-ONNX' if getattr(model, 'is_onnx', False) else 'RayKnight-41M'
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        tokens = line.strip().split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == 'uci':
            print(f'id name {engine_tag}')
            print('id author Antigravity & User')
            print('uciok')
            sys.stdout.flush()
        elif cmd == 'isready':
            print('readyok')
            sys.stdout.flush()
        elif cmd == 'ucinewgame':
            board.reset()
            history.clear()
        elif cmd == 'position':
            if len(tokens) > 1 and tokens[1] == 'startpos':
                board.reset()
                history.clear()
                if 'moves' in tokens:
                    idx = tokens.index('moves')
                    for m in tokens[idx + 1:]:
                        history.append(board.copy())
                        board.push_uci(m)
            elif len(tokens) > 1 and tokens[1] == 'fen':
                moves_idx = tokens.index('moves') if 'moves' in tokens else len(tokens)
                fen_str = ' '.join(tokens[2:moves_idx])
                board.set_fen(fen_str)
                history.clear()
                if moves_idx < len(tokens):
                    for m in tokens[moves_idx + 1:]:
                        history.append(board.copy())
                        board.push_uci(m)
        elif cmd == 'go':
            best_move, _, _ = mcts.search(board, history=history, num_simulations=100)
            if best_move:
                print(f'bestmove {best_move.uci()}')
            else:
                print('bestmove (none)')
            sys.stdout.flush()
        elif cmd == 'quit':
            break
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ray-Knight UCI Engine Interface')
    parser.add_argument('--model', type=str, default=None, help='Path to .pt or .onnx model checkpoint')
    args, remaining = parser.parse_known_args()
    model_arg = args.model or (remaining[0] if remaining else None)
    run_uci(model_path=model_arg)
