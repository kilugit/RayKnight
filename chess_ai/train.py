import argparse
import os
import re
import torch
from chess_ai.model import ChessNet, MODEL_PRESETS, load_model
from chess_ai.trainer import Trainer

def _parse_time_limit(limit_str: str | None) -> float | None:
    if not limit_str:
        return None
    s = limit_str.strip().lower()
    m = re.match('^(\\d+(?:\\.\\d+)?)\\s*([smh]?)$', s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return None
    val, unit = (float(m.group(1)), m.group(2))
    if unit == 'h':
        return val * 3600.0
    elif unit == 'm':
        return val * 60.0
    return val

def train_supervised(parquet_path: str='dataset/dataset.parquet', save_dir: str='checkpoints', preset: str='fast', batch_size: int=256, min_elo: int | None=2000, max_elo: int | None=None, lr: float | None=None, max_steps: int=1000, time_limit_sec: float | None=None, scheduler: str='wsd', val_every_steps: int=200, num_workers: int=4, shuffle_buffer_size: int=2048, resume_path: str | None=None, init_from: str | None=None, device: str='cuda' if torch.cuda.is_available() else 'cpu', num_blocks: int | None=None, num_filters: int | None=None, save_path: str | None=None) -> ChessNet:
    if save_path:
        save_dir = os.path.dirname(save_path) or save_dir
    if num_blocks is not None and num_filters is not None:
        config = {'num_blocks': num_blocks, 'num_filters': num_filters}
    else:
        config = MODEL_PRESETS.get(preset, MODEL_PRESETS['fast'])
    if lr is None:
        lr = 0.001 if config.get('num_filters', 128) <= 128 else 0.0003
    if init_from and os.path.exists(init_from):
        print(f'Initializing weights from: {init_from}')
        model, _ = load_model(init_from, device=device)
    else:
        model = ChessNet(**config).to(device)
    trainer = Trainer(model=model, parquet_path=parquet_path, save_dir=save_dir, batch_size=batch_size, lr=lr, min_elo=min_elo, max_elo=max_elo, num_workers=num_workers, shuffle_buffer_size=shuffle_buffer_size, device=device)
    trained_model = trainer.train(steps=max_steps, max_steps=max_steps, time_limit_sec=time_limit_sec, scheduler_type=scheduler, val_every_steps=val_every_steps, resume_checkpoint=resume_path)
    return trained_model
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ChessAI using SOTA training pipeline.')
    parser.add_argument('--dataset', '--parquet-path', dest='parquet_path', type=str, default='dataset/dataset.parquet', help='Path to parquet dataset file')
    parser.add_argument('--preset', choices=['fast', 'full'], default='fast', help='Model size preset (fast: 10x128, full: 32x256)')
    parser.add_argument('--scheduler', choices=['wsd', 'cosine', 'onecycle'], default='wsd', help='Learning rate schedule (wsd, cosine, onecycle)')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum training steps')
    parser.add_argument('--time-limit', type=str, default=None, help='Time limit (e.g. 4h, 30m, 3600s)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size per training step')
    parser.add_argument('--lr', type=float, default=None, help='Peak learning rate (default 1e-3 for fast, 3e-4 for full)')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader background workers')
    parser.add_argument('--min-elo', type=int, default=2000, help='Minimum player Elo')
    parser.add_argument('--max-elo', type=int, default=None, help='Maximum player Elo')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training state from')
    parser.add_argument('--init-from', type=str, default=None, help='Path to checkpoint to initialize model weights from')
    args = parser.parse_args()
    t_limit = _parse_time_limit(args.time_limit)
    train_supervised(parquet_path=args.parquet_path, preset=args.preset, batch_size=args.batch_size, lr=args.lr, max_steps=args.steps, time_limit_sec=t_limit, scheduler=args.scheduler, num_workers=args.workers, min_elo=args.min_elo, max_elo=args.max_elo, save_dir=args.save_dir, resume_path=args.resume, init_from=args.init_from)
