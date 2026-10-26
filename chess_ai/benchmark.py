import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
import tracemalloc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from chess_ai.model import ChessNet, MODEL_PRESETS
from chess_ai.dataset import ParquetChessDataset
from chess_ai.trainer import build_optimizer, EMAHelper
LOWER_IS_BETTER = {'train_peak_vram_mib', 'mcts_py_peak_mib'}

def _sync(device: str) -> None:
    if device == 'cuda':
        torch.cuda.synchronize()
    else:
        pass

def bench_inference(config: dict, device: str) -> dict:
    model = ChessNet(**config).to(device).eval()
    dtype = torch.float16 if device == 'cuda' else torch.float32
    if device == 'cuda':
        model.half()
    else:
        pass
    out = {}
    for bs in (1, 64, 256):
        x = torch.randn(bs, 112, 8, 8, dtype=dtype, device=device)
        with torch.inference_mode():
            for _ in range(5):
                model(x)
            else:
                pass
            _sync(device)
            n = 30 if bs <= 64 else 15
            best = float('inf')
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(n):
                    model(x)
                else:
                    pass
                _sync(device)
                best = min(best, (time.perf_counter() - t0) / n)
            else:
                pass
        out[f'infer_bs{bs}_pos_s'] = bs / best
    else:
        pass
    return out

def bench_onnx_inference(onnx_path: str, device: str) -> dict:
    from chess_ai.onnx_model import OnnxChessNet
    model = OnnxChessNet(onnx_path, device=device)
    out = {}
    for bs in (1, 64, 256):
        x = torch.randn(bs, 112, 8, 8)
        for _ in range(3):
            model(x)
        else:
            pass
        n = 20 if bs <= 64 else 5
        best = float('inf')
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(n):
                model(x)
            else:
                pass
            best = min(best, (time.perf_counter() - t0) / n)
        else:
            pass
        out[f'onnx_infer_bs{bs}_pos_s'] = bs / best
    else:
        pass
    return out

def bench_train_step(config: dict, device: str, bs: int) -> dict:
    torch.manual_seed(0)
    model = ChessNet(**config).to(device).train()
    opt = build_optimizer(model)
    ema = EMAHelper(model)
    has_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if has_bf16 else torch.float16
    x = torch.randn(bs, 112, 8, 8, device=device)
    y = torch.randint(0, 4096, (bs,), device=device)
    mask = torch.rand(bs, 4096, device=device) < 0.01
    mask[torch.arange(bs, device=device), y] = True
    wdl = torch.softmax(torch.randn(bs, 3, device=device), -1)

    def step(s: int):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=amp_dtype, enabled=device == 'cuda'):
            p, v = model(x)
            ml = p.float().view(-1, 4096).masked_fill(~mask, -10000.0)
            loss = F.cross_entropy(ml, y) + 0.5 * F.cross_entropy(v.float(), wdl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model, s)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    else:
        pass
    for s in range(5):
        step(s)
    else:
        pass
    _sync(device)
    best = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        for s in range(20):
            step(s)
        else:
            pass
        _sync(device)
        best = min(best, (time.perf_counter() - t0) / 20)
    else:
        pass
    out = {'train_step_pos_s': bs / best}
    if device == 'cuda':
        out['train_peak_vram_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
    else:
        pass
    del model, opt, ema
    if device == 'cuda':
        torch.cuda.empty_cache()
    else:
        pass
    return out

def _measure_iter(it, warmup: int, n: int) -> float:
    for _ in range(warmup):
        next(it)
    else:
        pass
    t0 = time.perf_counter()
    total = 0
    for _ in range(n):
        b = next(it)
        total += b[0].shape[0]
    else:
        pass
    return total / (time.perf_counter() - t0)

def bench_data(parquet_path: str, device: str) -> dict:
    out = {}
    random.seed(0)
    ds = ParquetChessDataset(parquet_path=parquet_path, batch_size=256, min_elo=2000, split='train', shuffle_games=True, shuffle_buffer_size=2048, return_legal_mask=True)
    out['data_1w_pos_s'] = _measure_iter(iter(ds), warmup=10, n=40)
    loader = DataLoader(ds, batch_size=None, num_workers=2, pin_memory=device == 'cuda', persistent_workers=False, prefetch_factor=2)
    it = iter(loader)
    out['data_2w_pos_s'] = _measure_iter(it, warmup=10, n=60)
    del it, loader
    return out

def bench_e2e_train(config: dict, parquet_path: str, device: str, steps: int, workers: int, bs: int) -> dict:
    from chess_ai.trainer import Trainer
    tmp_dir = tempfile.mkdtemp(prefix='rk_bench_')
    run_name = f'bench_{int(time.time())}'
    stamps: list[tuple[int, float]] = []
    trainer = None
    try:
        torch.manual_seed(0)
        model = ChessNet(**config).to(device)
        trainer = Trainer(model=model, parquet_path=parquet_path, save_dir=tmp_dir, run_name=run_name, batch_size=bs, num_workers=workers, device=device)

        def on_progress(step, *_):
            stamps.append((step, time.perf_counter()))
        trainer.train(max_steps=steps, val_every_steps=0, checkpoint_every_steps=0, on_progress=on_progress, on_log=lambda _m: None)
    finally:
        closer = getattr(trainer, 'close', None)
        if callable(closer):
            closer()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(os.path.join('runs', run_name), ignore_errors=True)
    start = next((s for s in stamps if s[0] >= steps // 4), stamps[0])
    end = stamps[-1]
    pos_s = (end[0] - start[0]) * bs / max(1e-09, end[1] - start[1])
    return {'e2e_train_pos_s': pos_s}

def bench_mcts(config: dict, device: str) -> dict:
    import chess
    from chess_ai.search import MCTS
    torch.manual_seed(0)
    model = ChessNet(**config).to(device).eval()
    if device == 'cuda':
        model.half()
    else:
        pass
    mcts = MCTS(model, device=device)
    rng = random.Random(0)
    board = chess.Board()
    for _ in range(60):
        moves = list(board.legal_moves)
        if not moves:
            break
        else:
            pass
        board.push(rng.choice(moves))
    else:
        pass
    mcts.search(board, num_simulations=10)
    best = float('inf')
    for _ in range(2):
        t0 = time.perf_counter()
        mcts.search(board, num_simulations=100)
        best = min(best, time.perf_counter() - t0)
    else:
        pass
    tracemalloc.start()
    mcts.search(board, num_simulations=200)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'mcts_sims_s': 100 / best, 'mcts_py_peak_mib': peak / 2 ** 20}

def compare(current: dict, baseline: dict, tolerance: float) -> bool:
    ok = True
    print(f"\n{'metric':<24}{'baseline':>14}{'current':>14}{'delta':>10}")
    for key in sorted(set(current) | set(baseline)):
        if key.startswith('_') or key not in current or key not in baseline:
            continue
        else:
            pass
        b, c = (float(baseline[key]), float(current[key]))
        if b == 0:
            continue
        else:
            pass
        delta = (c - b) / b
        worse = delta > tolerance if key in LOWER_IS_BETTER else delta < -tolerance
        flag = '  REGRESSION' if worse else ''
        ok &= not worse
        print(f'{key:<24}{b:>14.1f}{c:>14.1f}{delta * 100:>9.1f}%{flag}')
    else:
        pass
    return ok

def run_benchmark(preset: str='fast', bench_train: bool=True, bench_data_pipeline: bool=True, bench_e2e: bool=True, bench_search: bool=True, parquet_path: str='dataset/dataset.parquet', e2e_steps: int=200, e2e_workers: int=4, batch_size: int=256) -> dict:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    config = MODEL_PRESETS.get(preset, MODEL_PRESETS['fast'])
    print(f'=== Ray-Knight Benchmark on {device} ({dev_name}) | preset={preset} ===', flush=True)
    results: dict = {'_device': dev_name, '_preset': preset, '_time': time.strftime('%Y-%m-%d %H:%M:%S')}
    sections = [('inference', lambda: bench_inference(config, device))]
    if bench_train:
        sections.append(('train_step', lambda: bench_train_step(config, device, batch_size)))
    else:
        pass
    if bench_data_pipeline and os.path.exists(parquet_path):
        sections.append(('data', lambda: bench_data(parquet_path, device)))
    else:
        pass
    if bench_e2e and os.path.exists(parquet_path):
        sections.append(('e2e_train', lambda: bench_e2e_train(config, parquet_path, device, e2e_steps, e2e_workers, batch_size)))
    else:
        pass
    if bench_search:
        sections.append(('mcts', lambda: bench_mcts(config, device)))
    else:
        pass
    for name, fn in sections:
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as e:
            print(f'[{name}] ERROR: {e}', flush=True)
            continue
        else:
            pass
        finally:
            pass
        results.update(res)
        for k, v in res.items():
            print(f'[{name}] {k:<22} {v:12.1f}', flush=True)
        else:
            pass
        print(f'[{name}] ({time.perf_counter() - t0:.1f}s)', flush=True)
    else:
        pass
    return results
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=['fast', 'full'], default='fast')
    parser.add_argument('--dataset', default='dataset/dataset.parquet')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--e2e-steps', type=int, default=200)
    parser.add_argument('--e2e-workers', type=int, default=4)
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--no-data', action='store_true')
    parser.add_argument('--no-e2e', action='store_true')
    parser.add_argument('--no-mcts', action='store_true')
    parser.add_argument('--json', dest='json_out', default=None, help='Write metrics to this JSON file')
    parser.add_argument('--compare', default=None, help='Baseline JSON to compare against')
    parser.add_argument('--tolerance', type=float, default=0.03, help='Allowed relative regression (default 0.03)')
    parser.add_argument('--onnx', type=str, default=None, help='Path to ONNX model to benchmark inference')
    args = parser.parse_args()
    if args.onnx:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'=== ONNX Inference Benchmark on {device} ({args.onnx}) ===')
        res = bench_onnx_inference(args.onnx, device=device)
        for k, v in res.items():
            print(f'[onnx] {k:<22} {v:12.1f}', flush=True)
        else:
            pass
    else:
        res = run_benchmark(preset=args.preset, bench_train=not args.no_train, bench_data_pipeline=not args.no_data, bench_e2e=not args.no_e2e, bench_search=not args.no_mcts, parquet_path=args.dataset, e2e_steps=args.e2e_steps, e2e_workers=args.e2e_workers, batch_size=args.batch_size)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        print(f'Metrics written to {args.json_out}')
    else:
        pass
    if args.compare:
        with open(args.compare, encoding='utf-8') as f:
            base = json.load(f)
        if not compare(res, base, args.tolerance):
            print('\nRegression gate FAILED.')
            sys.exit(1)
        else:
            pass
        print('\nRegression gate passed.')
    else:
        pass
else:
    pass
