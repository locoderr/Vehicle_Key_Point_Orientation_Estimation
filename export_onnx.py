"""
Export the stage-2 KeyPointModel to ONNX and save preprocessing stats.

Produces two files next to the ONNX model:
  weights/stage2.onnx          — the model
  weights/stage2_mean_std.npz  — VeRi normalisation mean/std (for infer_onnx.py)

Usage (run from the landmark_detector directory):
    python export_onnx.py
    python export_onnx.py --ckpt weights/best_fine_kp_checkpoint.pth.tar --output weights/stage2.onnx
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser('Export stage-2 KeyPointModel to ONNX')
    p.add_argument('--ckpt', default='weights/best_fine_kp_checkpoint.pth.tar',
                   help='Path to stage-2 checkpoint')
    p.add_argument('--output', default='weights/stage2.onnx',
                   help='Output ONNX file path')
    p.add_argument('--mean_std_src', default='data/VeRi/mean.pth.tar',
                   help='VeRi mean/std .pth.tar file (input)')
    p.add_argument('--opset', default=17, type=int,
                   help='ONNX opset version (default: 17)')
    p.add_argument('--no_verify', action='store_true',
                   help='Skip onnxruntime numerical verification')
    return p.parse_args()


def export(args):
    from models.KP_Orientation_Net import KeyPointModel

    # Export on CPU — produces a portable model usable on any device
    device = torch.device('cpu')

    print('Loading checkpoint ...')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    net = KeyPointModel().to(device)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f'  Parameters: {n_params:,}')

    # Dummy inputs matching expected inference shapes
    dummy_224 = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    dummy_56  = torch.zeros(1, 3,  56,  56, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f'Exporting ONNX (opset {args.opset}) ->{args.output} ...')
    with torch.no_grad():
        # dynamo=False uses the stable TorchScript-based exporter,
        # which is simpler, well-tested, and accepts dynamic_axes directly.
        torch.onnx.export(
            net,
            (dummy_224, dummy_56),
            args.output,
            dynamo=False,
            opset_version=args.opset,
            input_names=['image_224', 'image_56'],
            output_names=['coarse_kp', 'fine_kp', 'orientation'],
            dynamic_axes={
                'image_224':   {0: 'batch'},
                'image_56':    {0: 'batch'},
                'coarse_kp':   {0: 'batch'},
                'fine_kp':     {0: 'batch'},
                'orientation': {0: 'batch'},
            },
        )
    print('  Export done.')

    # Validate graph structure with onnx library if available
    try:
        import onnx
        model_proto = onnx.load(args.output)
        onnx.checker.check_model(model_proto)
        print('  ONNX graph check: OK')
    except ImportError:
        print('  (onnx package not installed — skipping graph check)')

    # Save mean/std as .npz so the eval script needs no torch at all
    npz_path = os.path.splitext(args.output)[0] + '_mean_std.npz'
    ms = torch.load(args.mean_std_src, map_location='cpu', weights_only=False)
    mean = ms['mean'].numpy().astype(np.float64)
    std  = ms['std'].numpy().astype(np.float64)
    np.savez(npz_path, mean=mean, std=std)
    print(f'  mean/std saved ->{npz_path}')
    print(f'  mean = {mean}')
    print(f'  std  = {std}')

    # Numerical verification with onnxruntime
    if not args.no_verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])

            # Run torch forward
            with torch.no_grad():
                pt_coarse, pt_fine, pt_ori = net(dummy_224, dummy_56)

            # Run ONNX forward
            ort_coarse, ort_fine, ort_ori = sess.run(None, {
                'image_224': dummy_224.numpy(),
                'image_56':  dummy_56.numpy(),
            })

            def check(name, pt, ort_out):
                pt_np = pt.numpy()
                diff = np.abs(pt_np - ort_out).max()
                print(f'  {name:<15s} shape={tuple(ort_out.shape)}   max_diff={diff:.2e}  {"OK" if diff < 1e-4 else "WARN: large diff"}')

            print('Numerical verification (max abs diff vs PyTorch):')
            check('coarse_kp',  pt_coarse, ort_coarse)
            check('fine_kp',    pt_fine,   ort_fine)
            check('orientation', pt_ori,   ort_ori)

        except ImportError:
            print('  (onnxruntime not installed — skipping numerical check)')

    size_mb = os.path.getsize(args.output) / 1e6
    print(f'\nDone.  {args.output}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    export(parse_args())
