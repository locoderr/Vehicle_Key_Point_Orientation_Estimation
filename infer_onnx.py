"""
Standalone ONNX inference for the vehicle keypoint + orientation detector.

Dependencies: onnxruntime, opencv-python, numpy, matplotlib
NO PyTorch required.

Usage:
    python infer_onnx.py --image samples/sample.png
    python infer_onnx.py --image samples/sample.png --conf_thresh 0.4 --device cuda
"""
import argparse
import os
import sys

import cv2
import numpy as np

ORIENTATION_LABELS = [
    'front', 'rear', 'left', 'left front',
    'left rear', 'right', 'right front', 'right rear',
]
KP_LABELS = [
    'left-front wheel', 'left-back wheel', 'right-front wheel', 'right-back wheel',
    'right fog lamp', 'left fog lamp', 'right headlight', 'left headlight',
    'front auto logo', 'front license plate', 'left rear-view mirror',
    'right rear-view mirror', 'right-front corner of vehicle top',
    'left-front corner of vehicle top', 'left-back corner of vehicle top',
    'right-back corner of vehicle top', 'left rear lamp', 'right rear lamp',
    'rear auto logo', 'rear license plate',
]

# BGR palette for OpenCV drawing
_COLORS_BGR = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 0, 128), (0, 128, 0),
    (128, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0),
    (0, 128, 255), (128, 0, 255), (0, 255, 128), (128, 255, 0),
    (255, 0, 128), (255, 128, 0), (128, 128, 255), (128, 255, 128),
    (200, 200, 200),  # extra for background class if present
]


# ---------------------------------------------------------------------------
# Preprocessing (pure NumPy + OpenCV, no torch)
# ---------------------------------------------------------------------------

def load_mean_std(npz_path: str):
    data = np.load(npz_path)
    return data['mean'].astype(np.float64), data['std'].astype(np.float64)


def preprocess(image_path: str, mean: np.ndarray, std: np.ndarray):
    """
    Returns two float32 CHW arrays with a batch dimension:
      t224 : (1, 3, 224, 224)
      t56  : (1, 3,  56,  56)

    Steps mirror the original VeriDataset pipeline:
      1. Read as RGB uint8
      2. Cast to float64
      3. Subtract per-channel mean, divide by per-channel std
         (stats computed on the VeRi-776 training set, pixel range 0-255)
      4. Resize with bilinear interpolation
      5. HWC ->CHW, add batch dim, cast to float32
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        sys.exit(f'Cannot read image: {image_path}')

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    # Per-channel normalisation (VeRi statistics, pixel range 0-255)
    for c in range(3):
        rgb[:, :, c] = (rgb[:, :, c] - mean[c]) / std[c]

    # cv2.resize wants (W, H); INTER_LINEAR handles both up- and down-scaling
    img_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_56  = cv2.resize(rgb, (56,  56),  interpolation=cv2.INTER_LINEAR)

    t224 = img_224.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    t56  = img_56.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return t224, t56


# ---------------------------------------------------------------------------
# Post-processing (pure NumPy)
# ---------------------------------------------------------------------------

def decode_keypoints(fine_kp: np.ndarray, orig_h: int, orig_w: int):
    """
    fine_kp : (1, C, 56, 56) — normalised heatmaps from FineRegressor
    Returns  : list of (x_px, y_px, score) in original image coordinates
    """
    hm = fine_kp[0]  # (C, 56, 56)
    coords = []
    for c in range(hm.shape[0]):
        flat = hm[c].ravel()
        idx  = int(flat.argmax())
        x56  = idx % 56
        y56  = idx // 56
        coords.append((
            x56 * orig_w / 56,
            y56 * orig_h / 56,
            float(flat[idx]),
        ))
    return coords


# ---------------------------------------------------------------------------
# Visualisation (OpenCV)
# ---------------------------------------------------------------------------

def draw_and_save(image_path: str, coords, orientation_label: str,
                  save_path: str, conf_thresh: float = 0.0, radius: int = 8):
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]

    # Draw keypoints
    for i, (x, y, score) in enumerate(coords):
        if score < conf_thresh:
            continue
        cx, cy = int(round(x)), int(round(y))
        color = _COLORS_BGR[i % len(_COLORS_BGR)]
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(img, (cx, cy), radius, (255, 255, 255), 1)
        cv2.putText(img, f'#{i} {score:.3f}',
                    (cx + radius + 2, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Orientation banner at the top
    banner_h = 28
    banner = np.zeros((banner_h, orig_w, 3), dtype=np.uint8)
    cv2.putText(banner, f'Orientation: {orientation_label}',
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 180), 2, cv2.LINE_AA)

    # Legend panel on the right
    shown = [(i, s) for i, (_, _, s) in enumerate(coords) if s >= conf_thresh]
    legend_w = 240
    row_h    = 16
    legend_h = max(orig_h, len(shown) * row_h + 10)
    legend   = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)

    for row, (i, score) in enumerate(shown):
        y_pos = row * row_h + row_h
        color = _COLORS_BGR[i % len(_COLORS_BGR)]
        cv2.rectangle(legend, (4, y_pos - 9), (14, y_pos + 3), color, -1)
        label = KP_LABELS[i] if i < len(KP_LABELS) else '???'
        cv2.putText(legend, f'#{i} {label}',
                    (18, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (200, 200, 200), 1, cv2.LINE_AA)

    # Stack: banner on top of image, then legend on the right
    img_with_banner = np.vstack([banner, img])
    pad_h = img_with_banner.shape[0] - legend_h
    if pad_h > 0:
        legend = np.vstack([legend, np.zeros((pad_h, legend_w, 3), dtype=np.uint8)])
    else:
        img_with_banner = np.vstack([
            img_with_banner,
            np.zeros((-pad_h, orig_w, 3), dtype=np.uint8)
        ])
    out = np.hstack([img_with_banner, legend])

    cv2.imwrite(save_path, out)
    print(f'Result saved ->{save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser('ONNX vehicle landmark inference — no PyTorch')
    p.add_argument('--image',       default='samples/sample.png')
    p.add_argument('--model',       default='weights/stage2.onnx',
                   help='Path to exported ONNX model')
    p.add_argument('--mean_std',    default='weights/stage2_mean_std.npz',
                   help='Path to mean/std .npz saved by export_onnx.py')
    p.add_argument('--output',      default='',
                   help='Output image path (default: <image>_onnx_result.<ext>)')
    p.add_argument('--conf_thresh', default=0.0, type=float,
                   help='Confidence threshold: hide keypoints below this value')
    p.add_argument('--device',      default='cpu', choices=['cpu', 'cuda'],
                   help='Execution provider: cpu or cuda')
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in [(args.image,    'image'),
                        (args.model,    'ONNX model (run export_onnx.py first)'),
                        (args.mean_std, 'mean/std file (run export_onnx.py first)')]:
        if not os.path.isfile(path):
            sys.exit(f'{label} not found: {path}')

    # --- Load normalisation stats ---
    mean, std = load_mean_std(args.mean_std)

    # --- Preprocess ---
    orig_bgr = cv2.imread(args.image)
    orig_h, orig_w = orig_bgr.shape[:2]
    t224, t56 = preprocess(args.image, mean, std)

    # --- Build ONNX session ---
    import onnxruntime as ort
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if args.device == 'cuda' else ['CPUExecutionProvider'])
    sess = ort.InferenceSession(args.model, providers=providers)
    active_provider = sess.get_providers()[0]
    print(f'ONNX Runtime provider : {active_provider}')

    # --- Inference ---
    _coarse_kp, fine_kp, orientation_logits = sess.run(None, {
        'image_224': t224,
        'image_56':  t56,
    })

    # --- Decode orientation ---
    pred_cls   = int(np.argmax(orientation_logits[0]))
    ori_label  = ORIENTATION_LABELS[pred_cls]
    print(f'Predicted orientation : {ori_label}  (class {pred_cls})')

    # --- Decode keypoints ---
    coords = decode_keypoints(fine_kp, orig_h, orig_w)
    n_shown = sum(1 for _, _, s in coords if s >= args.conf_thresh)
    print(f'\nKeypoints  thresh={args.conf_thresh:.3f}  showing {n_shown}/{len(coords)}:')
    for i, (x, y, score) in enumerate(coords):
        label  = KP_LABELS[i] if i < len(KP_LABELS) else '???'
        suffix = '' if score >= args.conf_thresh else '  [filtered]'
        print(f'  [{i:2d}] {label:<40s}  x={int(round(x)):4d}  y={int(round(y)):4d}  score={score:.4f}{suffix}')

    # --- Save visualisation ---
    if not args.output:
        base, ext = os.path.splitext(args.image)
        args.output = base + '_onnx_result' + (ext or '.png')

    draw_and_save(args.image, coords, ori_label,
                  args.output, conf_thresh=args.conf_thresh)


if __name__ == '__main__':
    main()
