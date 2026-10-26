import math
import os
import sys
from typing import Any, Iterator
import numpy as np
import onnx
try:
    _null_fd = os.open(os.devnull, os.O_RDWR)
    _save_stderr = os.dup(2)
    os.dup2(_null_fd, 2)
    try:
        import onnxruntime as ort
    finally:
        os.dup2(_save_stderr, 2)
        os.close(_null_fd)
        os.close(_save_stderr)
except Exception:
    import onnxruntime as ort
try:
    ort.set_default_logger_severity_level(3)
except Exception:
    pass
import torch

def select_onnx_providers(device: str='cpu') -> list[str]:
    available = ort.get_available_providers()
    device_lower = str(device).lower()
    if 'cuda' in device_lower:
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers
    if device_lower in ('dml', 'directml', 'gpu'):
        providers = []
        if 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers
    return ['CPUExecutionProvider']

class OnnxChessNet:
    is_onnx: bool = True

    def __init__(self, model_path: str, device: str='cpu', providers: list[str] | None=None, session_options: ort.SessionOptions | None=None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'ONNX model not found: {model_path}')
        self.path = os.path.abspath(model_path)
        self.device = str(device)
        if session_options is None:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
        else:
            opts = session_options
        chosen_providers = providers or select_onnx_providers(self.device)
        try:
            self.session = ort.InferenceSession(self.path, opts, providers=chosen_providers)
        except Exception:
            self.session = ort.InferenceSession(self.path, opts, providers=['CPUExecutionProvider'])
        self.active_provider = self.session.get_providers()[0]
        inputs = self.session.get_inputs()
        if not inputs:
            raise ValueError(f'ONNX model at {model_path} has no inputs.')
        self.input_name = inputs[0].name
        self.input_type = inputs[0].type
        outputs = self.session.get_outputs()
        if len(outputs) < 2:
            raise ValueError(f'ONNX model expected at least 2 outputs (policy, value), got {len(outputs)}')
        self.policy_idx: int | None = None
        self.value_idx: int | None = None
        for idx, out in enumerate(outputs):
            name_lower = out.name.lower()
            if 'policy' in name_lower or 'pol' in name_lower:
                self.policy_idx = idx
            elif 'value' in name_lower or 'val' in name_lower:
                self.value_idx = idx
        if self.policy_idx is None or self.value_idx is None:
            for idx, out in enumerate(outputs):
                shape = out.shape
                if shape and len(shape) >= 1:
                    last_dim = shape[-1]
                    if last_dim == 3 and self.value_idx is None:
                        self.value_idx = idx
                    elif last_dim in (64, 4096) and self.policy_idx is None:
                        self.policy_idx = idx
        if self.policy_idx is None:
            self.policy_idx = 0
        if self.value_idx is None:
            self.value_idx = 1
        self._param_count: int | None = None
        self.num_blocks: int = 10
        self.num_filters: int = 128
        self._inspect_onnx_metadata()

    def _inspect_onnx_metadata(self) -> None:
        try:
            proto = onnx.load(self.path, load_external_data=False)
            total_params = 0
            block_indices = set()
            found_filters = None
            for init in proto.graph.initializer:
                dims = list(init.dims)
                total_params += math.prod(dims) if dims else 0
                if 'embedding.weight' in init.name and dims:
                    found_filters = dims[0]
                if 'body.' in init.name:
                    parts = init.name.split('body.')[1].split('.')
                    if parts and parts[0].isdigit():
                        block_indices.add(int(parts[0]))
            self._param_count = total_params
            if found_filters is not None:
                self.num_filters = found_filters
            if block_indices:
                self.num_blocks = max(block_indices) + 1
        except Exception:
            self._param_count = None

    def count_parameters(self) -> int:
        if self._param_count is not None:
            return self._param_count
        return 0

    def get_config(self) -> dict[str, Any]:
        return {'format': 'onnx', 'path': self.path, 'num_blocks': self.num_blocks, 'num_filters': self.num_filters, 'active_provider': self.active_provider}

    def eval(self) -> 'OnnxChessNet':
        return self

    def train(self, mode: bool=False) -> 'OnnxChessNet':
        if mode:
            import warnings
            warnings.warn('OnnxChessNet is an inference-only model and cannot be set to training mode.')
        return self

    def half(self) -> 'OnnxChessNet':
        return self

    def float(self) -> 'OnnxChessNet':
        return self

    def to(self, device: str) -> 'OnnxChessNet':
        device_str = str(device)
        if device_str != self.device:
            new_providers = select_onnx_providers(device_str)
            if new_providers != self.session.get_providers():
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                try:
                    self.session = ort.InferenceSession(self.path, opts, providers=new_providers)
                    self.active_provider = self.session.get_providers()[0]
                except Exception:
                    pass
            self.device = device_str
        return self

    def parameters(self) -> Iterator[torch.Tensor]:
        dtype = torch.float16 if 'float16' in self.input_type else torch.float32
        device_target = self.device if self.device != 'dml' else 'cpu'
        try:
            dummy = torch.empty(0, dtype=dtype, device=device_target)
        except Exception:
            dummy = torch.empty(0, dtype=dtype, device='cpu')
        return iter([dummy])

    def forward(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return self(x)

    def __call__(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            orig_device = x.device
            x_np = x.detach().cpu().numpy()
        else:
            orig_device = torch.device(self.device if self.device != 'dml' else 'cpu')
            x_np = np.asarray(x)
        if 'float16' in self.input_type:
            if x_np.dtype != np.float16:
                x_np = x_np.astype(np.float16)
        elif x_np.dtype != np.float32:
            x_np = x_np.astype(np.float32)
        single_input = x_np.ndim == 3
        if single_input:
            x_np = np.expand_dims(x_np, axis=0)
        raw_outputs = self.session.run(None, {self.input_name: x_np})
        p_raw = raw_outputs[self.policy_idx]
        v_raw = raw_outputs[self.value_idx]
        p_t = torch.from_numpy(p_raw).to(orig_device)
        v_t = torch.from_numpy(v_raw).to(orig_device)
        if p_t.ndim == 2 and p_t.shape[-1] == 4096:
            p_t = p_t.view(-1, 64, 64)
        if single_input:
            p_t = p_t.squeeze(0)
            v_t = v_t.squeeze(0)
        return (p_t, v_t)
