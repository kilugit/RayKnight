import torch
from chess_ai.model import RayKnightOperator, ChessNet
from tests.reference_rko import ReferenceRayKnightOperator

def test_ray_knight_operator_forward_equivalence():
    torch.manual_seed(42)
    in_c, out_c = (64, 64)
    ref = ReferenceRayKnightOperator(in_c, out_c)
    fast = RayKnightOperator(in_c, out_c)
    fast.load_state_dict(ref.state_dict())
    x = torch.randn(4, in_c, 8, 8, dtype=torch.float32)
    with torch.no_grad():
        out_ref = ref(x)
        out_fast = fast(x)
    diff = (out_ref - out_fast).abs().max().item()
    print('Forward pass max diff:', diff)
    assert diff < 1e-05, f'Discrepancy {diff} exceeds tolerance 1e-5'

def test_ray_knight_operator_backward_equivalence():
    torch.manual_seed(42)
    in_c, out_c = (32, 32)
    ref = ReferenceRayKnightOperator(in_c, out_c)
    fast = RayKnightOperator(in_c, out_c)
    fast.load_state_dict(ref.state_dict())
    x_ref = torch.randn(2, in_c, 8, 8, requires_grad=True)
    x_fast = x_ref.clone().detach().requires_grad_(True)
    out_ref = ref(x_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    out_fast = fast(x_fast)
    loss_fast = out_fast.sum()
    loss_fast.backward()
    grad_in_diff = (x_ref.grad - x_fast.grad).abs().max().item()
    assert grad_in_diff < 0.0001, f'Input grad diff {grad_in_diff} exceeds tolerance'
    for (name_ref, p_ref), (name_fast, p_fast) in zip(ref.named_parameters(), fast.named_parameters()):
        assert name_ref == name_fast
        assert p_ref.grad is not None and p_fast.grad is not None
        p_diff = (p_ref.grad - p_fast.grad).abs().max().item()
        assert p_diff < 0.0001, f'Param {name_ref} grad diff {p_diff} exceeds tolerance'
