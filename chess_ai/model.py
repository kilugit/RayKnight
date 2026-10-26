from __future__ import annotations
import os
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
if TYPE_CHECKING:
    from chess_ai.onnx_model import OnnxChessNet

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        var = torch.mean(x_f32.pow(2), dim=1, keepdim=True)
        norm = x_f32 * torch.rsqrt(var + self.eps)
        w = self.weight.view(1, -1, 1, 1)
        return (norm * w).to(orig_dtype)

class SqueezeExciteBlock(nn.Module):

    def __init__(self, channels: int=256, se_ratio: int=8):
        super().__init__()
        reduced = max(1, channels // se_ratio)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sigmoid(self.fc2(F.silu(self.fc1(self.gap(x)))))
        return x * w

def _diag_shifts(z: torch.Tensor, dist: int) -> list[torch.Tensor]:
    return [F.pad(z[:, :, :-dist, :-dist], (dist, 0, dist, 0)), F.pad(z[:, :, :-dist, dist:], (0, dist, dist, 0)), F.pad(z[:, :, dist:, :-dist], (dist, 0, 0, dist)), F.pad(z[:, :, dist:, dist:], (0, dist, 0, dist))]

def _knight_shifts(z: torch.Tensor) -> list[torch.Tensor]:
    return [F.pad(z[:, :, :-2, :-1], (1, 0, 2, 0)), F.pad(z[:, :, :-2, 1:], (0, 1, 2, 0)), F.pad(z[:, :, 2:, :-1], (1, 0, 0, 2)), F.pad(z[:, :, 2:, 1:], (0, 1, 0, 2)), F.pad(z[:, :, :-1, :-2], (2, 0, 1, 0)), F.pad(z[:, :, :-1, 2:], (0, 2, 1, 0)), F.pad(z[:, :, 1:, :-2], (2, 0, 0, 1)), F.pad(z[:, :, 1:, 2:], (0, 2, 0, 1))]

def _build_shift_bases() -> tuple[torch.Tensor, torch.Tensor]:
    eye = torch.eye(64).view(64, 1, 8, 8)
    diag = torch.zeros(4, 7, 64, 64)
    for dist in range(1, 8):
        for k, s in enumerate(_diag_shifts(eye, dist)):
            diag[k, dist - 1] = s.view(64, 64)
    knight = torch.stack([s.view(64, 64) for s in _knight_shifts(eye)])
    return (diag.view(28, 64 * 64), knight.view(8, 64 * 64))
_DIAG_BASIS, _KNIGHT_BASIS = _build_shift_bases()

class RayKnightOperator(nn.Module):

    def __init__(self, in_channels: int=256, out_channels: int=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_rank = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_file = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.ray_dist_weights = nn.Parameter(torch.randn(8, 7) * 0.05)
        self.knight_weights = nn.Parameter(torch.randn(8) * 0.05)
        self.knight_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_out = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.register_buffer('diag_basis', _DIAG_BASIS.clone(), persistent=False)
        self.register_buffer('knight_basis', _KNIGHT_BASIS.clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.conv_in(x))
        b, c = z.shape[:2]
        zf = z.reshape(b, c, 64)
        ray_ortho = self.conv_rank(z) + self.conv_file(z)
        diag_w = self.ray_dist_weights[4:8].reshape(1, 28)
        m_diag = (diag_w.to(self.diag_basis.dtype) @ self.diag_basis).view(64, 64)
        ray_diag = (zf @ m_diag.to(zf.dtype)).view_as(z)
        ray_out = ray_ortho + ray_diag
        m_knight = (self.knight_weights.view(1, 8).to(self.knight_basis.dtype) @ self.knight_basis).view(64, 64)
        knight_out = self.knight_proj((zf @ m_knight.to(zf.dtype)).view_as(z))
        combined = torch.cat([ray_out, knight_out], dim=1)
        return F.silu(self.conv_out(combined))

class RB_SE_RK(nn.Module):

    def __init__(self, channels: int=256, se_ratio: int=8):
        super().__init__()
        self.rko = RayKnightOperator(channels, channels)
        self.norm1 = RMSNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = RMSNorm(channels)
        self.se = SqueezeExciteBlock(channels, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(self.rko(x))
        x = self.norm2(self.conv2(x))
        x = self.se(x)
        return F.silu(x + residual)
MODEL_PRESETS: dict[str, dict[str, int]] = {'fast': {'num_blocks': 10, 'num_filters': 128}, 'full': {'num_blocks': 32, 'num_filters': 256}}

class ChessNet(nn.Module):

    def __init__(self, num_blocks: int=10, num_filters: int=128):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.embedding = nn.Conv2d(112, num_filters, kernel_size=3, padding=1, bias=False)
        self.norm_embed = RMSNorm(num_filters)
        self.body = nn.Sequential(*[RB_SE_RK(num_filters) for _ in range(num_blocks)])
        self.policy_conv = nn.Conv2d(num_filters, 64, kernel_size=1)
        self.policy_src = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1), nn.SiLU(), nn.Conv2d(128, 1, kernel_size=1))
        self.policy_dst = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1), nn.SiLU(), nn.Conv2d(128, 1, kernel_size=1))
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_mlp = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, 160), nn.SiLU(), nn.Linear(160, 3))

    def get_config(self) -> dict[str, int]:
        return {'num_blocks': self.num_blocks, 'num_filters': self.num_filters}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.norm_embed(self.embedding(x)))
        x = self.body(x)
        p = F.silu(self.policy_conv(x))
        src = self.policy_src(p).view(-1, 64)
        dst = self.policy_dst(p).view(-1, 64)
        policy_logits = src.unsqueeze(-1) + dst.unsqueeze(-2)
        v = F.silu(self.value_conv(x))
        value_logits = self.value_mlp(v)
        return (policy_logits, value_logits)

    def count_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

def mask_illegal_moves(policy_logits: torch.Tensor, legal_mask: torch.Tensor, mask_value: float=-10000.0) -> torch.Tensor:
    masked = policy_logits.clone()
    masked[~legal_mask] = mask_value
    return masked

def save_checkpoint(path: str, model: ChessNet, *, config: dict[str, int] | None=None, ema_state: dict[str, torch.Tensor] | None=None, optimizer: torch.optim.Optimizer | None=None, scheduler: object | None=None, step: int=0, metrics: dict | None=None) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = f'{path}.tmp'
    payload = {'format_version': 2, 'model_config': config or model.get_config(), 'model_state': {k: v.cpu() for k, v in model.state_dict().items()}, 'ema_state': {k: v.cpu() for k, v in ema_state.items()} if ema_state is not None else None, 'optimizer_state': optimizer.state_dict() if optimizer is not None else None, 'scheduler_state': scheduler.state_dict() if scheduler is not None and hasattr(scheduler, 'state_dict') else None, 'step': step, 'metrics': metrics or {}}
    torch.save(payload, tmp_path)
    if os.path.exists(path):
        try:
            os.replace(tmp_path, path)
        except OSError:
            os.remove(path)
            os.rename(tmp_path, path)
    else:
        os.rename(tmp_path, path)

def infer_model_config_from_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, int]:
    if 'embedding.weight' in sd:
        num_filters = sd['embedding.weight'].shape[0]
    else:
        num_filters = 128
    block_indices = set()
    for key in sd.keys():
        if key.startswith('body.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    num_blocks = max(block_indices) + 1 if block_indices else 10
    return {'num_blocks': num_blocks, 'num_filters': num_filters}

def _is_valid_onnx_model(path: str) -> bool:
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        import onnx
        proto = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(proto)
        return True
    except Exception:
        return False

def load_model(checkpoint_path_or_dict: str | dict, device: str='cpu', prefer_ema: bool=True, fallback_config: dict[str, int] | None=None) -> tuple[ChessNet | OnnxChessNet, dict]:
    if isinstance(checkpoint_path_or_dict, str):
        if checkpoint_path_or_dict.endswith('.onnx') or _is_valid_onnx_model(checkpoint_path_or_dict):
            from chess_ai.onnx_model import OnnxChessNet
            onnx_model = OnnxChessNet(checkpoint_path_or_dict, device=device)
            meta = {'format_version': 'onnx', 'model_config': onnx_model.get_config(), 'step': 0, 'metrics': {}, 'is_onnx': True}
            return (onnx_model, meta)
        data = torch.load(checkpoint_path_or_dict, map_location='cpu')
    else:
        data = checkpoint_path_or_dict
    if isinstance(data, dict) and data.get('format_version') == 2:
        config = data.get('model_config') or fallback_config or MODEL_PRESETS['fast']
        model = ChessNet(**config).to(device)
        weights = data['ema_state'] if prefer_ema and data.get('ema_state') is not None else data['model_state']
        model.load_state_dict(weights, strict=True)
        return (model, data)
    sd = data
    if isinstance(sd, dict):
        config = infer_model_config_from_state_dict(sd)
        model = ChessNet(**config).to(device)
        model.load_state_dict(sd, strict=True)
        meta = {'format_version': 1, 'model_config': config, 'step': 0, 'metrics': {}}
        return (model, meta)
    raise ValueError(f'Unrecognized checkpoint format: {type(data)}')

def _is_valid_checkpoint(path: str) -> bool:
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        if path.endswith('.onnx'):
            return _is_valid_onnx_model(path)
        sd = torch.load(path, map_location='cpu', mmap=True)
        if isinstance(sd, dict):
            if sd.get('format_version') == 2:
                model_state = sd.get('model_state', {})
                return bool(model_state)
            return bool(sd)
        return False
    except Exception:
        return False

def find_latest_checkpoint(checkpoint_dir: str='checkpoints', include_onnx: bool=True) -> str | None:
    if not os.path.exists(checkpoint_dir):
        return None
    priority_names = ['best_model.pt', 'model.pt']
    if include_onnx:
        priority_names.append('model.onnx')
    for name in priority_names:
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path) and _is_valid_checkpoint(path):
            return path
    valid_exts = ('.pt', '.onnx') if include_onnx else ('.pt',)
    candidates = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(valid_exts) and os.path.isfile(os.path.join(checkpoint_dir, f))]
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        for f in candidates:
            if _is_valid_checkpoint(f):
                return f
    return None
