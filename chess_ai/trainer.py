import csv
import math
import os
import time
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from chess_ai.dataset import ParquetChessDataset
from chess_ai.encoder import unpack_states_gpu, unpack_legal_masks_gpu
from chess_ai.model import ChessNet, MODEL_PRESETS, save_checkpoint, load_model

def _worker_init_fn(worker_id: int) -> None:
    torch.set_num_threads(1)

class WSDScheduler:

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, min_lr: float=1e-05, total_steps: int | None=1000, time_limit_sec: float | None=None, warmup_ratio: float=0.03, decay_ratio: float=0.2):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.time_limit_sec = time_limit_sec
        self.warmup_ratio = warmup_ratio
        self.decay_ratio = decay_ratio
        self.start_time = time.time()
        self.current_step = 0

    def get_progress(self, step: int) -> float:
        if self.time_limit_sec and self.time_limit_sec > 0:
            elapsed = time.time() - self.start_time
            time_prog = min(1.0, elapsed / self.time_limit_sec)
            if self.total_steps and self.total_steps > 0:
                step_prog = min(1.0, step / self.total_steps)
                return max(time_prog, step_prog)
            return time_prog
        if self.total_steps and self.total_steps > 0:
            return min(1.0, step / self.total_steps)
        return 0.0

    def step(self, step: int) -> float:
        self.current_step = step
        prog = self.get_progress(step)
        if prog < self.warmup_ratio:
            lr = self.min_lr + (self.base_lr - self.min_lr) * (prog / max(1e-06, self.warmup_ratio))
        elif prog < 1.0 - self.decay_ratio:
            lr = self.base_lr
        else:
            decay_prog = (prog - (1.0 - self.decay_ratio)) / max(1e-06, self.decay_ratio)
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * decay_prog))
            lr = self.min_lr + (self.base_lr - self.min_lr) * decay_factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def state_dict(self) -> dict:
        return {'current_step': self.current_step, 'start_time': self.start_time, 'base_lr': self.base_lr, 'min_lr': self.min_lr}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state.get('current_step', 0)
        self.base_lr = state.get('base_lr', self.base_lr)
        self.min_lr = state.get('min_lr', self.min_lr)

class CosineScheduler:

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, min_lr: float=1e-05, total_steps: int | None=1000, time_limit_sec: float | None=None, warmup_ratio: float=0.05):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.time_limit_sec = time_limit_sec
        self.warmup_ratio = warmup_ratio
        self.start_time = time.time()
        self.current_step = 0

    def get_progress(self, step: int) -> float:
        if self.time_limit_sec and self.time_limit_sec > 0:
            elapsed = time.time() - self.start_time
            time_prog = min(1.0, elapsed / self.time_limit_sec)
            if self.total_steps and self.total_steps > 0:
                step_prog = min(1.0, step / self.total_steps)
                return max(time_prog, step_prog)
            return time_prog
        if self.total_steps and self.total_steps > 0:
            return min(1.0, step / self.total_steps)
        return 0.0

    def step(self, step: int) -> float:
        self.current_step = step
        prog = self.get_progress(step)
        if prog < self.warmup_ratio:
            lr = self.min_lr + (self.base_lr - self.min_lr) * (prog / max(1e-06, self.warmup_ratio))
        else:
            decay_prog = (prog - self.warmup_ratio) / max(1e-06, 1.0 - self.warmup_ratio)
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * decay_prog))
            lr = self.min_lr + (self.base_lr - self.min_lr) * decay_factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def state_dict(self) -> dict:
        return {'current_step': self.current_step, 'start_time': self.start_time, 'base_lr': self.base_lr, 'min_lr': self.min_lr}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state.get('current_step', 0)
        self.base_lr = state.get('base_lr', self.base_lr)
        self.min_lr = state.get('min_lr', self.min_lr)

class OneCycleScheduler:

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, total_steps: int | None=1000, time_limit_sec: float | None=None, pct_start: float=0.3, div_factor: float=25.0, final_div_factor: float=1000.0):
        self.optimizer = optimizer
        self.max_lr = base_lr
        self.initial_lr = base_lr / div_factor
        self.min_lr = base_lr / final_div_factor
        self.total_steps = total_steps
        self.time_limit_sec = time_limit_sec
        self.pct_start = pct_start
        self.start_time = time.time()
        self.current_step = 0

    def get_progress(self, step: int) -> float:
        if self.time_limit_sec and self.time_limit_sec > 0:
            elapsed = time.time() - self.start_time
            time_prog = min(1.0, elapsed / self.time_limit_sec)
            if self.total_steps and self.total_steps > 0:
                step_prog = min(1.0, step / self.total_steps)
                return max(time_prog, step_prog)
            return time_prog
        if self.total_steps and self.total_steps > 0:
            return min(1.0, step / self.total_steps)
        return 0.0

    def step(self, step: int) -> float:
        self.current_step = step
        prog = self.get_progress(step)
        if prog < self.pct_start:
            phase_prog = prog / max(1e-06, self.pct_start)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * phase_prog
        else:
            phase_prog = (prog - self.pct_start) / max(1e-06, 1.0 - self.pct_start)
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * phase_prog))
            lr = self.min_lr + (self.max_lr - self.min_lr) * decay_factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def state_dict(self) -> dict:
        return {'current_step': self.current_step, 'start_time': self.start_time, 'max_lr': self.max_lr}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state.get('current_step', 0)
        self.max_lr = state.get('max_lr', self.max_lr)

def get_lr_scheduler(scheduler_name: str, optimizer: torch.optim.Optimizer, base_lr: float, total_steps: int | None=1000, time_limit_sec: float | None=None):
    name = (scheduler_name or 'wsd').lower()
    if name == 'cosine':
        return CosineScheduler(optimizer, base_lr=base_lr, total_steps=total_steps, time_limit_sec=time_limit_sec)
    elif name in ['onecycle', 'one_cycle']:
        return OneCycleScheduler(optimizer, base_lr=base_lr, total_steps=total_steps, time_limit_sec=time_limit_sec)
    else:
        return WSDScheduler(optimizer, base_lr=base_lr, total_steps=total_steps, time_limit_sec=time_limit_sec)

def build_optimizer(model: ChessNet, lr: float=0.001, weight_decay: float=0.0001, fused: bool | None=None) -> AdamW:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and (not any((nd in name for nd in ['norm', 'bias', 'ray_dist', 'knight_weights']))):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': no_decay_params, 'weight_decay': 0.0}]
    if fused is None:
        try:
            p0 = next(model.parameters(), None)
            fused = p0 is not None and p0.is_cuda
        except Exception:
            fused = False
    try:
        return AdamW(groups, lr=lr, fused=fused)
    except Exception:
        return AdamW(groups, lr=lr, fused=False)

class EMAHelper:

    def __init__(self, model: ChessNet, decay: float=0.999, device: torch.device | str | None=None):
        self.decay = decay
        p0 = next(model.parameters(), None)
        self.device = device or (p0.device if p0 is not None else torch.device('cpu'))
        self.shadow: dict[str, torch.Tensor] = {k: v.clone().detach().to(self.device) for k, v in model.state_dict().items()}
        self._param_names = [k for k, _ in model.named_parameters()]
        self._param_refs = [p for p in model.parameters()]
        self._shadow_param_refs = [self.shadow[k] for k in self._param_names if k in self.shadow]

    def update(self, model: ChessNet, step: int) -> None:
        cur_decay = min(self.decay, (1.0 + step) / (10.0 + step))
        weight = 1.0 - cur_decay
        if self._param_refs and self._shadow_param_refs:
            torch._foreach_lerp_(self._shadow_param_refs, self._param_refs, weight)
        else:
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].lerp_(v.detach().to(self.device), weight)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone().detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.device))
            else:
                self.shadow[k] = v.clone().detach().to(self.device)
        self._shadow_param_refs = [self.shadow[k] for k in self._param_names if k in self.shadow]

    def apply_to(self, target_model: ChessNet) -> None:
        target_model.load_state_dict(self.shadow)

class Trainer:

    def __init__(self, model: ChessNet, parquet_path: str='dataset/dataset.parquet', save_dir: str='checkpoints', run_name: str | None=None, batch_size: int=256, lr: float=0.001, weight_decay: float=0.0001, value_weight: float=0.5, label_smoothing: float=0.05, ema_decay: float=0.999, min_elo: int | None=2000, max_elo: int | None=None, num_workers: int=0, shuffle_buffer_size: int=2048, device: str | None=None, resume: bool | str | None=None, resume_checkpoint: str | None=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.parquet_path = parquet_path
        if resume_checkpoint:
            self.resume_checkpoint: str | None = resume_checkpoint
        elif isinstance(resume, str) and resume:
            self.resume_checkpoint = resume
        elif resume is True:
            self.resume_checkpoint = os.path.join(self.save_dir, 'last.pt')
        else:
            self.resume_checkpoint = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.float().to(self.device)
        self.is_cuda = self.device == 'cuda'
        self.has_bf16 = self.is_cuda and torch.cuda.is_bf16_supported()
        self.amp_dtype = torch.bfloat16 if self.has_bf16 else torch.float16
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.is_cuda and (not self.has_bf16))
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.value_weight = value_weight
        self.label_smoothing = label_smoothing
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.optimizer = build_optimizer(self.model, lr=lr, weight_decay=weight_decay)
        self.ema = EMAHelper(self.model, decay=ema_decay)
        self._eval_model: ChessNet | None = None
        self._cached_val_batches: list[tuple] | None = None
        self._csv_file = None
        self._csv_writer = None
        self.run_id = run_name or time.strftime('run_%Y%m%d_%H%M%S')
        self.log_dir = os.path.join('runs', self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None
        if self._eval_model is not None:
            self._eval_model = None
        if self._cached_val_batches is not None:
            self._cached_val_batches.clear()
            self._cached_val_batches = None
        if hasattr(self, 'ema') and self.ema is not None:
            self.ema.shadow.clear()
            if hasattr(self.ema, '_shadow_param_refs'):
                self.ema._shadow_param_refs.clear()
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.state.clear()
        import gc
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()

    def _init_csv(self):
        if self._csv_file is None:
            file_exists = os.path.exists(self.csv_path)
            self._csv_file = open(self.csv_path, 'a', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_file)
            if not file_exists or os.path.getsize(self.csv_path) == 0:
                self._csv_writer.writerow(['step', 'time_sec', 'lr', 'loss', 'loss_policy', 'loss_val', 'top1_acc', 'val_loss_policy', 'val_top1', 'val_loss_val', 'samples_sec'])
                self._csv_file.flush()

    def _log_csv(self, row: dict):
        if self._csv_writer is None:
            self._init_csv()
        self._csv_writer.writerow([row.get('step', 0), f"{row.get('time_sec', 0.0):.2f}", f"{row.get('lr', 0.0):.6f}", f"{row.get('loss', 0.0):.4f}", f"{row.get('loss_policy', 0.0):.4f}", f"{row.get('loss_val', 0.0):.4f}", f"{row.get('top1_acc', 0.0):.4f}", f"{row.get('val_loss_policy', ''):.4f}" if isinstance(row.get('val_loss_policy'), float) else '', f"{row.get('val_top1', ''):.4f}" if isinstance(row.get('val_top1'), float) else '', f"{row.get('val_loss_val', ''):.4f}" if isinstance(row.get('val_loss_val'), float) else '', f"{row.get('samples_sec', 0.0):.1f}"])

    def evaluate(self, num_val_batches: int=15) -> dict[str, float]:
        if self._eval_model is None:
            self._eval_model = ChessNet(**self.model.get_config()).to(self.device).eval()
        self.ema.apply_to(self._eval_model)
        if self._cached_val_batches is None:
            val_ds = ParquetChessDataset(parquet_path=self.parquet_path, batch_size=self.batch_size, min_elo=self.min_elo, max_elo=self.max_elo, split='val', shuffle_games=False, return_legal_mask=True, output_format='packed' if self.is_cuda else 'dense')
            cached = []
            for batch in val_ds:
                cached.append(batch)
                if len(cached) >= num_val_batches:
                    break
            self._cached_val_batches = cached
        total_loss_p = 0.0
        total_loss_v = 0.0
        total_top1 = 0.0
        batches = 0
        with torch.inference_mode():
            for batch in self._cached_val_batches:
                if len(batch) == 5 and batch[0].ndim == 3:
                    pieces_bb, scalars, target_policies, target_wdl, legal_masks_bb = batch
                    pieces_bb = pieces_bb.to(self.device, non_blocking=True)
                    scalars = scalars.to(self.device, non_blocking=True)
                    target_policies = target_policies.to(self.device, non_blocking=True)
                    target_wdl = target_wdl.to(self.device, non_blocking=True)
                    legal_masks_bb = legal_masks_bb.to(self.device, non_blocking=True)
                    states = unpack_states_gpu(pieces_bb, scalars)
                    legal_masks = unpack_legal_masks_gpu(legal_masks_bb)
                else:
                    states, target_policies, target_wdl, legal_masks = batch
                    states = states.to(self.device, dtype=torch.float32 if not self.is_cuda else None, non_blocking=True)
                    target_policies = target_policies.to(self.device, non_blocking=True)
                    target_wdl = target_wdl.to(self.device, non_blocking=True)
                    legal_masks = legal_masks.to(self.device, non_blocking=True)
                with torch.autocast(self.device, dtype=self.amp_dtype, enabled=self.is_cuda):
                    p_logits, v_logits = self._eval_model(states)
                    masked_p = p_logits.float().view(-1, 4096).masked_fill(~legal_masks, -10000.0)
                    lp = F.cross_entropy(masked_p, target_policies)
                    lv = F.cross_entropy(v_logits.float(), target_wdl)
                top1 = (masked_p.argmax(dim=-1) == target_policies).float().mean().item()
                total_loss_p += lp.item()
                total_loss_v += lv.item()
                total_top1 += top1
                batches += 1
        if batches == 0:
            return {'val_loss_policy': float('nan'), 'val_top1': 0.0, 'val_loss_val': float('nan')}
        return {'val_loss_policy': total_loss_p / batches, 'val_top1': total_top1 / batches, 'val_loss_val': total_loss_v / batches}

    def train(self, max_steps: int | None=None, steps: int | None=None, time_limit_sec: float | None=None, scheduler_type: str='wsd', val_every_steps: int=200, checkpoint_every_steps: int=250, resume_checkpoint: str | None=None, on_progress: Callable[[int, int, float, float, float, float, float, float], None] | None=None, on_log: Callable[[str], None] | None=None, should_stop: Callable[[], bool] | None=None) -> ChessNet:
        self._init_csv()
        start_step = 1
        best_val_loss = float('inf')
        if resume_checkpoint is None and self.resume_checkpoint is not None:
            resume_checkpoint = self.resume_checkpoint
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            try:
                test_data = torch.load(resume_checkpoint, map_location='cpu')
                nan_found = False
                for st_key in ['model_state', 'ema_state']:
                    st_dict = test_data.get(st_key)
                    if st_dict and any((torch.isnan(v).any().item() for v in st_dict.values())):
                        nan_found = True
                        break
                if nan_found:
                    fallback_best = os.path.join(self.save_dir, 'best_model.pt')
                    if os.path.exists(fallback_best) and fallback_best != resume_checkpoint:
                        msg = f"Resume checkpoint '{resume_checkpoint}' contains corrupted non-finite weights! Falling back to '{fallback_best}'."
                        if on_log:
                            on_log(msg)
                        else:
                            print(msg)
                        resume_checkpoint = fallback_best
            except Exception:
                pass
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            if on_log:
                on_log(f'Resuming from checkpoint: {resume_checkpoint}')
            loaded_model, meta = load_model(resume_checkpoint, device=self.device)
            self.model.load_state_dict(loaded_model.state_dict())
            start_step = meta.get('step', 0) + 1
            if meta.get('optimizer_state'):
                try:
                    self.optimizer.load_state_dict(meta['optimizer_state'])
                except Exception:
                    pass
            if meta.get('ema_state'):
                self.ema.load_state_dict(meta['ema_state'])
        elif resume_checkpoint:
            msg = f"Resume checkpoint '{resume_checkpoint}' not found; starting from current weights."
            if on_log:
                on_log(msg)
            else:
                print(msg)
        if steps is not None:
            total_session_steps = steps
            target_step = start_step + steps - 1
        elif max_steps is not None:
            if max_steps >= start_step:
                target_step = max_steps
                total_session_steps = max_steps - start_step + 1
            else:
                total_session_steps = max_steps
                target_step = start_step + max_steps - 1
                if on_log:
                    on_log(f'Resuming: requested {max_steps} steps (training steps {start_step} to {target_step}).')
        else:
            total_session_steps = 1000
            target_step = start_step + 1000 - 1
        scheduler = get_lr_scheduler(scheduler_name=scheduler_type, optimizer=self.optimizer, base_lr=self.lr, total_steps=total_session_steps, time_limit_sec=time_limit_sec)
        train_ds = ParquetChessDataset(parquet_path=self.parquet_path, batch_size=self.batch_size, min_elo=self.min_elo, max_elo=self.max_elo, split='train', shuffle_games=True, shuffle_buffer_size=self.shuffle_buffer_size, return_legal_mask=True, output_format='packed' if self.is_cuda else 'dense')
        if self.num_workers > 0:
            loader = DataLoader(train_ds, batch_size=None, num_workers=self.num_workers, pin_memory=self.is_cuda, persistent_workers=True, prefetch_factor=2, worker_init_fn=_worker_init_fn)
        else:
            loader = train_ds
        data_iter = iter(loader)
        self.model.train()
        train_start = time.time()
        last_flush_time = train_start
        samples_processed = 0
        prec_desc = 'bf16' if self.has_bf16 else 'fp16' if self.is_cuda else 'fp32'
        msg = f'Training started on {self.device} ({prec_desc}) | {self.model.count_parameters():,} params'
        if on_log:
            on_log(msg)
        else:
            print(msg)
        step = start_step
        consecutive_nans = 0
        pending_stats = []
        flush_every = 10
        val_metrics = {}

        def _flush_stats():
            nonlocal pending_stats, last_flush_time, consecutive_nans
            if not pending_stats:
                return
            tensors = torch.stack([torch.stack([p[2], p[3], p[4], p[5]]) for p in pending_stats])
            data_cpu = tensors.cpu().tolist()
            now = time.time()
            elapsed_window = now - last_flush_time
            total_samples = sum((p[6] for p in pending_stats))
            samples_sec = total_samples / elapsed_window if elapsed_window > 0 else 0.0
            last_flush_time = now
            latest_loss = 0.0
            latest_lp = 0.0
            latest_lv = 0.0
            latest_top1 = 0.0
            latest_lr = pending_stats[-1][1]
            latest_step = pending_stats[-1][0]
            for (s, lr_val, _, _, _, _, _, t_time), (l_val, lp_val, lv_val, top1_val) in zip(pending_stats, data_cpu):
                if math.isnan(l_val):
                    consecutive_nans += 1
                else:
                    consecutive_nans = 0
                latest_loss = l_val
                latest_lp = lp_val
                latest_lv = lv_val
                latest_top1 = top1_val
                self._log_csv({'step': s, 'time_sec': t_time - train_start, 'lr': lr_val, 'loss': l_val, 'loss_policy': lp_val, 'loss_val': lv_val, 'top1_acc': top1_val, 'val_loss_policy': val_metrics.get('val_loss_policy'), 'val_top1': val_metrics.get('val_top1'), 'val_loss_val': val_metrics.get('val_loss_val'), 'samples_sec': samples_sec})
            if on_progress:
                on_progress(latest_step, target_step, latest_loss, latest_lp, latest_lv, latest_top1, latest_lr, samples_sec)
            pending_stats.clear()
        try:
            while True:
                if should_stop and should_stop():
                    if on_log:
                        on_log('Training stopped by user request.')
                    break
                if target_step and step > target_step:
                    break
                if time_limit_sec and time.time() - train_start >= time_limit_sec:
                    if on_log:
                        on_log(f'Time limit of {time_limit_sec:.0f}s reached.')
                    break
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                if len(batch) == 5 and batch[0].ndim == 3:
                    pieces_bb, scalars, target_policies, target_wdl, legal_masks_bb = batch
                    pieces_bb = pieces_bb.to(self.device, non_blocking=True)
                    scalars = scalars.to(self.device, non_blocking=True)
                    target_policies = target_policies.to(self.device, non_blocking=True)
                    target_wdl = target_wdl.to(self.device, non_blocking=True)
                    legal_masks_bb = legal_masks_bb.to(self.device, non_blocking=True)
                    states = unpack_states_gpu(pieces_bb, scalars)
                    legal_masks = unpack_legal_masks_gpu(legal_masks_bb)
                    bs = pieces_bb.shape[0]
                else:
                    states, target_policies, target_wdl, legal_masks = batch
                    states = states.to(self.device, dtype=torch.float32 if not self.is_cuda else None, non_blocking=True)
                    target_policies = target_policies.to(self.device, non_blocking=True)
                    target_wdl = target_wdl.to(self.device, non_blocking=True)
                    legal_masks = legal_masks.to(self.device, non_blocking=True)
                    bs = states.shape[0]
                samples_processed += bs
                session_step = step - start_step + 1
                cur_lr = scheduler.step(session_step)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(self.device, dtype=self.amp_dtype, enabled=self.is_cuda):
                    p_logits, v_logits = self.model(states)
                    masked_logits = p_logits.float().view(-1, 4096).masked_fill(~legal_masks, -10000.0)
                    if self.label_smoothing > 0.0:
                        log_probs = F.log_softmax(masked_logits, dim=-1)
                        target_log_p = log_probs.gather(1, target_policies.unsqueeze(1)).squeeze(1).clamp(min=-100.0)
                        num_legals = legal_masks.sum(dim=-1).clamp_min(1.0)
                        sum_legal_log_p = log_probs.masked_fill(~legal_masks, 0.0).sum(dim=-1)
                        mean_legal_log_p = (sum_legal_log_p / num_legals).clamp(min=-100.0)
                        loss_policy = (-(1.0 - self.label_smoothing) * target_log_p - self.label_smoothing * mean_legal_log_p).mean()
                    else:
                        loss_policy = F.cross_entropy(masked_logits, target_policies)
                    loss_val = F.cross_entropy(v_logits.float(), target_wdl)
                    loss = loss_policy + self.value_weight * loss_val
                if not torch.isfinite(loss):
                    consecutive_nans += 1
                    if on_log:
                        on_log(f'Warning: Non-finite loss at step {step} (loss={loss.item()}). Skipping backward step.')
                    pending_stats.append((step, cur_lr, loss.detach(), loss_policy.detach(), loss_val.detach(), torch.tensor(0.0, device=self.device), bs, time.time()))
                    if len(pending_stats) >= flush_every:
                        _flush_stats()
                    if consecutive_nans >= 5:
                        if on_log:
                            on_log('Error: 5 consecutive NaN losses detected. Halting training session.')
                        break
                    step += 1
                    continue
                step_ok = True
                if self.is_cuda and (not self.has_bf16):
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if torch.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        step_ok = False
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                    else:
                        step_ok = False
                        self.optimizer.zero_grad(set_to_none=True)
                if step_ok:
                    consecutive_nans = 0
                    self.ema.update(self.model, step)
                else:
                    consecutive_nans += 1
                    if on_log:
                        on_log(f'Warning: Non-finite gradient norm ({grad_norm.item()}) at step {step}. Skipped parameter update to protect weights.')
                top1_t = (masked_logits.argmax(dim=-1) == target_policies).float().mean()
                now_step = time.time()
                pending_stats.append((step, cur_lr, loss.detach(), loss_policy.detach(), loss_val.detach(), top1_t.detach(), bs, now_step))
                if len(pending_stats) >= flush_every:
                    _flush_stats()
                    if consecutive_nans >= 5:
                        if on_log:
                            on_log('Error: 5 consecutive NaN losses detected. Halting training session.')
                        break
                if step % 25 == 0 or step == target_step:
                    _flush_stats()
                    log_line = f'Step {step}/{target_step} (session: {session_step}/{total_session_steps}) | Loss: {loss.item():.4f} (Pol: {loss_policy.item():.4f}, Val: {loss_val.item():.4f}) | Top-1: {top1_t.item() * 100:.1f}% | LR: {cur_lr:.2e}' if start_step > 1 else f'Step {step}/{target_step} | Loss: {loss.item():.4f} (Pol: {loss_policy.item():.4f}, Val: {loss_val.item():.4f}) | Top-1: {top1_t.item() * 100:.1f}% | LR: {cur_lr:.2e}'
                    if on_log:
                        on_log(log_line)
                    else:
                        print(log_line)
                if val_every_steps and (step % val_every_steps == 0 or step == target_step):
                    _flush_stats()
                    val_metrics = self.evaluate(num_val_batches=15)
                    val_lp = val_metrics['val_loss_policy']
                    val_acc = val_metrics['val_top1']
                    v_msg = f"[Validation @ Step {step}] Pol Loss: {val_lp:.4f} | Top-1: {val_acc * 100:.1f}% | Val Loss: {val_metrics['val_loss_val']:.4f}"
                    if on_log:
                        on_log(v_msg)
                    else:
                        print(v_msg)
                    if not math.isnan(val_lp) and val_lp < best_val_loss:
                        best_val_loss = val_lp
                        best_path = os.path.join(self.save_dir, 'best_model.pt')
                        save_checkpoint(best_path, self.model, ema_state=self.ema.state_dict(), step=step, metrics=val_metrics)
                        if on_log:
                            on_log(f'-> Saved new best model to {best_path} (Val Pol Loss: {val_lp:.4f})')
                if checkpoint_every_steps and (step % checkpoint_every_steps == 0 or step == target_step):
                    _flush_stats()
                    model_finite = all((torch.isfinite(p).all().item() for p in self.model.parameters()))
                    if model_finite:
                        last_path = os.path.join(self.save_dir, 'last.pt')
                        save_checkpoint(last_path, self.model, ema_state=self.ema.state_dict(), optimizer=self.optimizer, scheduler=scheduler, step=step, metrics={'loss': loss.item() if torch.isfinite(loss) else 0.0, 'top1': top1_t.item(), **val_metrics})
                step += 1
            _flush_stats()
        finally:
            del data_iter
            if self.num_workers > 0:
                del loader
            if self._csv_file is not None:
                try:
                    self._csv_file.flush()
                except Exception:
                    pass
        final_path = os.path.join(self.save_dir, 'last.pt')
        model_finite = all((torch.isfinite(p).all().item() for p in self.model.parameters()))
        if model_finite:
            save_checkpoint(final_path, self.model, ema_state=self.ema.state_dict(), optimizer=self.optimizer, scheduler=scheduler, step=step - 1)
            if on_log:
                on_log(f'Training session ended. Checkpoint safely saved to {final_path} (step {step - 1}).')
        elif on_log:
            on_log(f'Training session ended with non-finite weights; preserved existing healthy {final_path}.')
        self.ema.apply_to(self.model)
        self.model.eval()
        return self.model
