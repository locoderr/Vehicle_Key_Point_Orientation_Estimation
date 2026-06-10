"""
Single-image inference for the vehicle keypoint + orientation detector.

Usage (run from the landmark_detector directory):
    python infer.py --image samples/sample.png
    python infer.py --image samples/sample.png --stage1_only
"""
import argparse
import os
import sys

import numpy as np
import torch
from skimage import io
from skimage import transform as sk_transform

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

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def load_mean_std(mean_std_path='data/VeRi/mean.pth.tar'):
    data = torch.load(mean_std_path, map_location='cpu', weights_only=False)
    return data['mean'].numpy(), data['std'].numpy()


def preprocess(image_path, mean, std):
    """Load and preprocess one image into two tensors (224x224 and 56x56)."""
    image = io.imread(image_path).astype(np.float64)

    # Handle grayscale
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    # Drop alpha channel
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Normalize with VeRi dataset statistics
    for j in range(3):
        image[:, :, j] = (image[:, :, j] - mean[j]) / std[j]

    img_224 = sk_transform.resize(image, (224, 224), anti_aliasing=True)
    img_56 = sk_transform.resize(image, (56, 56), anti_aliasing=True)

    # HWC -> CHW, add batch dim
    t224 = torch.from_numpy(img_224.transpose(2, 0, 1)).float().unsqueeze(0)
    t56 = torch.from_numpy(img_56.transpose(2, 0, 1)).float().unsqueeze(0)
    return t224, t56


def get_keypoint_coords(heatmaps, orig_h, orig_w):
    """Extract (x, y) pixel coords in original image space from 56x56 heatmaps."""
    # heatmaps: 1 x C x 56 x 56
    hm = heatmaps[0]  # C x 56 x 56
    coords = []
    for c in range(hm.shape[0]):
        flat = hm[c].reshape(-1)
        idx = int(flat.argmax())
        x56 = idx % 56
        y56 = idx // 56
        # scale back to original resolution
        x = x56 * orig_w / 56
        y = y56 * orig_h / 56
        coords.append((x, y, float(flat[idx])))
    return coords


def draw_keypoints(image_path, coords, save_path, radius=6, conf_thresh=0.0):
    """Overlay keypoint dots on the original image and save."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        img = io.imread(image_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)

        patches = []
        for i, (x, y, score) in enumerate(coords):
            if score < conf_thresh:
                continue
            color = tuple(c / 255.0 for c in COLORS[i % len(COLORS)])
            circle = plt.Circle((x, y), radius, color=color, fill=True, linewidth=1.5)
            ax.add_patch(circle)
            # Index + score as text next to the dot
            ax.text(x + radius + 2, y, f'#{i} {score:.3f}',
                    color='white', fontsize=5.5,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.7, linewidth=0))
            label = KP_LABELS[i] if i < len(KP_LABELS) else '???'
            patches.append(mpatches.Patch(color=color, label=f'#{i} {label}'))

        ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left',
                  fontsize=6, borderaxespad=0)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Result saved to: {save_path}')
    except ImportError:
        print('matplotlib not available — skipping visualization')


def main():
    parser = argparse.ArgumentParser('Vehicle Landmark Detector — single image inference')
    parser.add_argument('--image', default='samples/sample.png',
                        help='Path to input image')
    parser.add_argument('--stage1_ckpt', default='weights/best_checkpoint.pth.tar',
                        help='Path to stage-1 (CoarseRegressor) checkpoint')
    parser.add_argument('--stage2_ckpt', default='weights/best_fine_kp_checkpoint.pth.tar',
                        help='Path to stage-2 (KeyPointModel) checkpoint')
    parser.add_argument('--stage1_only', action='store_true',
                        help='Run only stage 1 (CoarseRegressor) inference')
    parser.add_argument('--output', default='',
                        help='Output image path (default: <image_dir>/result_<name>)')
    parser.add_argument('--mean_std', default='data/VeRi/mean.pth.tar',
                        help='Path to VeRi mean/std .pth.tar file')
    parser.add_argument('--conf_thresh', default=0.0, type=float,
                        help='Confidence threshold: keypoints below this score are hidden (default: 0.0 = show all)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f'Image not found: {args.image}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Add project root to path so imports work from any CWD
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.KP_Orientation_Net import CoarseRegressor, KeyPointModel

    # --- Load mean/std ---
    mean, std = load_mean_std(args.mean_std)

    # --- Preprocess ---
    orig_img = io.imread(args.image)
    orig_h, orig_w = orig_img.shape[:2]
    t224, t56 = preprocess(args.image, mean, std)
    t224, t56 = t224.to(device), t56.to(device)

    # --- Load model and run ---
    if args.stage1_only:
        print('Loading stage-1 checkpoint ...')
        ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        net = CoarseRegressor().to(device)
        net.load_state_dict(ckpt['net_state_dict'])
        net.eval()
        with torch.no_grad():
            coarse_kp = net(t224)
        fine_kp = coarse_kp[:, :20]
        orientation_idx = None
        print('Stage-1 inference complete.')
    else:
        print('Loading stage-2 checkpoint ...')
        ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
        net = KeyPointModel().to(device)
        net.load_state_dict(ckpt['net_state_dict'])
        net.eval()
        with torch.no_grad():
            coarse_kp, fine_kp, orientation = net(t224, t56)
        _, pred_cls = torch.max(orientation, 1)
        orientation_idx = pred_cls.item()
        print(f'Stage-2 inference complete.')
        print(f'Predicted orientation: {ORIENTATION_LABELS[orientation_idx]} (class {orientation_idx})')

    # --- Extract keypoint coords (on the 56x56 heatmap grid, then scale) ---
    coords = get_keypoint_coords(fine_kp.cpu(), orig_h, orig_w)
    n_visible = sum(1 for _, _, s in coords if s >= args.conf_thresh)
    print(f'\nKeypoint predictions (x, y, confidence) — thresh={args.conf_thresh:.3f}, showing {n_visible}/{len(coords)}:')
    for i, (x, y, score) in enumerate(coords):
        label = KP_LABELS[i] if i < len(KP_LABELS) else '???'
        marker = '' if score >= args.conf_thresh else ' [filtered]'
        print(f'  [{i:2d}] {label:<40s} x={x:4f}  y={y:4f}  score={score:.4f}{marker}')

    # --- Save result ---
    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(args.image)
        out_path = base + '_result' + (ext if ext else '.png')

    draw_keypoints(args.image, coords, out_path, conf_thresh=args.conf_thresh)


if __name__ == '__main__':
    main()
