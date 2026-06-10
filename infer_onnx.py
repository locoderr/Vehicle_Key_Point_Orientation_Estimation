"""
Standalone ONNX inference CLI for the vehicle keypoint + orientation detector.
No PyTorch required.

Usage:
    python infer_onnx.py --image samples/sample.png
    python infer_onnx.py --image samples/sample.png --conf_thresh 0.4 --device cuda
"""
import argparse
import os
import sys

import cv2

from vehicle_landmark_detector import VehicleLandmarkDetector


def parse_args():
    p = argparse.ArgumentParser('ONNX vehicle landmark inference -- no PyTorch')
    p.add_argument('--image',       default='samples/sample.png')
    p.add_argument('--model',       default='weights/stage2.onnx',
                   help='Path to exported ONNX model')
    p.add_argument('--mean_std',    default='weights/stage2_mean_std.npz',
                   help='Path to mean/std .npz saved by export_onnx.py')
    p.add_argument('--output',      default='',
                   help='Output image path (default: <image>_onnx_result.<ext>)')
    p.add_argument('--conf_thresh', default=0.0, type=float,
                   help='Hide keypoints below this confidence score')
    p.add_argument('--device',      default='cuda', choices=['cpu', 'cuda'],
                   help='Execution provider: cpu or cuda')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f'Image not found: {args.image}')

    det = VehicleLandmarkDetector(
        model_path    = args.model,
        mean_std_path = args.mean_std,
        device        = args.device,
        conf_thresh   = args.conf_thresh,
    )
    print(det)

    result = det.detect(args.image)

    # Print
    print(f'Predicted orientation : {result.orientation}  (class {result.orientation_idx})')
    visible = result.visible_keypoints(args.conf_thresh)
    n_total = len(result.keypoints)
    print(f'\nKeypoints  thresh={args.conf_thresh:.3f}  showing {len(visible)}/{n_total}:')
    for kp in result.keypoints:
        suffix = '' if kp.score >= args.conf_thresh else '  [filtered]'
        print(f'  [{kp.id:2d}] {kp.label:<40s}  '
              f'x={int(round(kp.x)):4d}  y={int(round(kp.y)):4d}  '
              f'score={kp.score:.4f}{suffix}')

    # Save visualisation
    vis = det.draw(args.image, result)
    if not args.output:
        base, ext = os.path.splitext(args.image)
        args.output = base + '_onnx_result' + (ext or '.png')
    cv2.imwrite(args.output, vis)
    print(f'Result saved ->{args.output}')


if __name__ == '__main__':
    main()
