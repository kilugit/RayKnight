import os
import chess
import pytest
import torch
from chess_ai.model import load_model, find_latest_checkpoint, _is_valid_checkpoint
from chess_ai.onnx_model import OnnxChessNet, select_onnx_providers
from chess_ai.search import MCTS

@pytest.fixture
def onnx_path():
    path = 'checkpoints/model.onnx'
    if not os.path.isfile(path):
        pytest.skip('checkpoints/model.onnx not found for ONNX tests')
    return path

def test_select_onnx_providers():
    cpu_providers = select_onnx_providers('cpu')
    assert 'CPUExecutionProvider' in cpu_providers
    cuda_providers = select_onnx_providers('cuda')
    assert 'CPUExecutionProvider' in cuda_providers

def test_load_onnx_model(onnx_path):
    model, meta = load_model(onnx_path, device='cpu')
    assert isinstance(model, OnnxChessNet)
    assert meta['format_version'] == 'onnx'
    assert meta['is_onnx'] is True
    assert model.is_onnx is True
    assert model.count_parameters() > 3000000
    assert model.num_blocks == 10
    assert model.num_filters == 128

def test_onnx_forward_pass_shapes_and_types(onnx_path):
    model, _ = load_model(onnx_path, device='cpu')
    x1 = torch.randn(112, 8, 8)
    p1, v1 = model(x1)
    assert isinstance(p1, torch.Tensor)
    assert isinstance(v1, torch.Tensor)
    assert p1.shape == (64, 64)
    assert v1.shape == (3,)
    x4 = torch.randn(4, 112, 8, 8, dtype=torch.float32)
    p4, v4 = model(x4)
    assert p4.shape == (4, 64, 64)
    assert v4.shape == (4, 3)
    x_half = torch.randn(2, 112, 8, 8, dtype=torch.float16)
    p_half, v_half = model(x_half)
    assert p_half.shape == (2, 64, 64)
    assert v_half.shape == (2, 3)

def test_onnx_compatibility_methods(onnx_path):
    model, _ = load_model(onnx_path, device='cpu')
    assert model.eval() is model
    assert model.train(False) is model
    assert model.half() is model
    assert model.float() is model
    assert model.to('cpu') is model
    params = list(model.parameters())
    assert len(params) == 1
    assert params[0].dtype in (torch.float16, torch.float32)
    cfg = model.get_config()
    assert cfg['format'] == 'onnx'
    assert cfg['num_blocks'] == 10
    assert cfg['num_filters'] == 128

def test_mcts_search_with_onnx(onnx_path):
    model, _ = load_model(onnx_path, device='cpu')
    mcts = MCTS(model, device='cpu')
    board = chess.Board()
    best_move, policy, wdl = mcts.search(board, num_simulations=10)
    assert best_move is not None
    assert best_move in board.legal_moves
    assert len(wdl) == 3
    assert abs(sum(wdl) - 1.0) < 0.001
    assert len(policy) > 0

def test_find_latest_checkpoint_onnx_priority(tmp_path):
    ckpt_dir = str(tmp_path / 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    assert find_latest_checkpoint(ckpt_dir) is None
    onnx_file = os.path.join(ckpt_dir, 'model.onnx')
    if os.path.isfile('checkpoints/model.onnx'):
        import shutil
        shutil.copyfile('checkpoints/model.onnx', onnx_file)
    else:
        from chess_ai.model import ChessNet
        from chess_ai.export_onnx import export_onnx
        net = ChessNet(num_blocks=1, num_filters=16)
        export_onnx(net, onnx_file)
    assert find_latest_checkpoint(ckpt_dir) == onnx_file
    pt_file = os.path.join(ckpt_dir, 'model.pt')
    torch.save({'weight': torch.tensor([1.0])}, pt_file)
    assert find_latest_checkpoint(ckpt_dir) == pt_file
    best_pt = os.path.join(ckpt_dir, 'best_model.pt')
    torch.save({'weight': torch.tensor([2.0])}, best_pt)
    assert find_latest_checkpoint(ckpt_dir) == best_pt

def test_invalid_and_corrupt_onnx_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        OnnxChessNet(str(tmp_path / 'missing.onnx'))
    corrupt_file = str(tmp_path / 'corrupt.onnx')
    with open(corrupt_file, 'wb') as f:
        f.write(b'not an onnx file at all')
    assert _is_valid_checkpoint(corrupt_file) is False

def test_model_import_does_not_load_onnxruntime():
    import subprocess
    import sys
    code = "import sys, chess_ai.model; assert 'onnxruntime' not in sys.modules; print('CLEAN_IMPORT')"
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert res.returncode == 0
    assert 'CLEAN_IMPORT' in res.stdout
    assert 'Init provider bridge failed' not in res.stderr
