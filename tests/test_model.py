import torch
from chess_ai.model import ChessNet, mask_illegal_moves, find_latest_checkpoint
from chess_ai.export_onnx import export_onnx
import os
import onnx

def test_model_parameter_count():
    model = ChessNet(num_blocks=32, num_filters=256)
    params = model.count_parameters()
    print(f'Total params: {params:,}')
    assert 38000000 <= params <= 42000000, f'Params {params} outside [38M, 42M]'

def test_model_forward_and_shapes():
    model = ChessNet(num_blocks=2, num_filters=64)
    x = torch.randn(2, 112, 8, 8)
    policy_logits, value_logits = model(x)
    assert policy_logits.shape == (2, 64, 64)
    assert value_logits.shape == (2, 3)

def test_model_fp16_execution():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessNet(num_blocks=2, num_filters=64).to(device).half()
    x = torch.randn(2, 112, 8, 8, dtype=torch.float16, device=device)
    policy_logits, value_logits = model(x)
    assert policy_logits.dtype == torch.float16
    assert value_logits.dtype == torch.float16
    assert not torch.isnan(policy_logits).any()
    assert not torch.isnan(value_logits).any()

def test_mask_illegal_moves():
    logits = torch.zeros(64, 64)
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[12, 28] = True
    masked = mask_illegal_moves(logits, mask)
    assert masked[12, 28].item() == 0.0
    assert masked[0, 0].item() == -10000.0

def test_onnx_export_and_check(tmp_path):
    model = ChessNet(num_blocks=2, num_filters=64)
    out_file = str(tmp_path / 'model.onnx')
    res = export_onnx(model, output_path=out_file, half_precision=False)
    assert os.path.exists(res)
    proto = onnx.load(res)
    onnx.checker.check_model(proto)

def test_find_latest_checkpoint(tmp_path):
    ckpt_dir = str(tmp_path / 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    assert find_latest_checkpoint(ckpt_dir) is None
    p1 = os.path.join(ckpt_dir, 'model.pt')
    torch.save({'weight': torch.tensor([1.0])}, p1)
    assert find_latest_checkpoint(ckpt_dir) == p1
    p2 = os.path.join(ckpt_dir, 'best_model.pt')
    torch.save({'weight': torch.tensor([2.0])}, p2)
    assert find_latest_checkpoint(ckpt_dir) == p2
    p_nan = os.path.join(ckpt_dir, 'corrupt.pt')
    torch.save({'weight': torch.tensor([float('nan')])}, p_nan)
    assert find_latest_checkpoint(ckpt_dir) == p2
