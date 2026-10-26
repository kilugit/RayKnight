import argparse
import os
import torch
import onnx
from chess_ai.model import ChessNet, MODEL_PRESETS, load_model

def export_onnx(model: ChessNet, output_path: str='checkpoints/model.onnx', half_precision: bool=False, opset_version: int=18) -> str:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    import copy
    device = 'cpu'
    export_model = copy.deepcopy(model).to(device).eval()
    if half_precision:
        export_model = export_model.half()
        dummy_input = torch.randn(1, 112, 8, 8, dtype=torch.float16, device=device)
    else:
        export_model = export_model.float()
        dummy_input = torch.randn(1, 112, 8, 8, dtype=torch.float32, device=device)
    torch.onnx.export(export_model, dummy_input, output_path, input_names=['board_state'], output_names=['policy_logits', 'value_logits'], dynamic_axes={'board_state': {0: 'batch_size'}, 'policy_logits': {0: 'batch_size'}, 'value_logits': {0: 'batch_size'}}, opset_version=opset_version, dynamo=False)
    onnx_proto = onnx.load(output_path)
    onnx.checker.check_model(onnx_proto)
    return output_path
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export ChessNet to ONNX format')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt', help='Path to .pt checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/model.onnx', help='Path to output .onnx file')
    parser.add_argument('--half', action='store_true', help='Export in FP16 precision')
    args = parser.parse_args()
    if os.path.exists(args.checkpoint):
        net, meta = load_model(args.checkpoint, device='cpu')
        print(f'Loaded checkpoint from {args.checkpoint} ({net.count_parameters():,} params)')
    else:
        net = ChessNet(**MODEL_PRESETS['fast'])
        print(f'Warning: Checkpoint {args.checkpoint} not found. Exporting initialized architecture.')
    out = export_onnx(net, output_path=args.output, half_precision=args.half)
    print(f'ONNX model successfully exported and verified: {out}')
