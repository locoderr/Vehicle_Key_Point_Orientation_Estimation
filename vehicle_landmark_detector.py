"""
Self-contained vehicle landmark detector based on ONNX Runtime.
No PyTorch required. Drop this single file into any project.

Dependencies: onnxruntime, opencv-python (cv2), numpy

Quick start
-----------
    from vehicle_landmark_detector import VehicleLandmarkDetector

    det = VehicleLandmarkDetector(
        model_path    = 'weights/stage2.onnx',
        mean_std_path = 'weights/stage2_mean_std.npz',
        device        = 'cpu',      # or 'cuda'
        conf_thresh   = 0.4,
    )

    # bgr_frame is a numpy (H, W, 3) uint8 array from cv2.VideoCapture / cv2.imread
    result = det.detect(bgr_frame)

    print(result.orientation)                       # e.g. 'left rear'
    for kp in result.visible_keypoints():           # filtered by conf_thresh
        print(kp.label, kp.x, kp.y, kp.score)

    vis = det.draw(bgr_frame, result)               # annotated copy, ready for display
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORIENTATION_LABELS: List[str] = [
    'front', 'rear', 'left', 'left front',
    'left rear', 'right', 'right front', 'right rear',
]

KEYPOINT_LABELS: List[str] = [
    'left-front wheel',               # 0
    'left-back wheel',                # 1
    'right-front wheel',              # 2
    'right-back wheel',               # 3
    'right fog lamp',                 # 4
    'left fog lamp',                  # 5
    'right headlight',                # 6
    'left headlight',                 # 7
    'front auto logo',                # 8
    'front license plate',            # 9
    'left rear-view mirror',          # 10
    'right rear-view mirror',         # 11
    'right-front corner of roof',     # 12
    'left-front corner of roof',      # 13
    'left-back corner of roof',       # 14
    'right-back corner of roof',      # 15
    'left rear lamp',                 # 16
    'right rear lamp',                # 17
    'rear auto logo',                 # 18
    'rear license plate',             # 19
]

# BGR palette — one colour per keypoint index
_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (0, 0, 255),   (0, 255, 0),   (255, 0, 0),   (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 0, 128),   (0, 128, 0),
    (128, 0, 0),   (0, 128, 128), (128, 0, 128), (128, 128, 0),
    (0, 128, 255), (128, 0, 255), (0, 255, 128), (128, 255, 0),
    (255, 0, 128), (255, 128, 0), (128, 128, 255),(128, 255, 128),
    (200, 200, 200),  # fallback for any extra channel
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Keypoint:
    """Single keypoint prediction in original image pixel coordinates."""
    id: int
    label: str
    x: float   # column in the source image
    y: float   # row    in the source image
    score: float


@dataclass
class LandmarkDetection:
    """Full result for one vehicle crop / frame."""
    orientation: str          # human-readable orientation label
    orientation_idx: int      # class index 0-7
    orientation_scores: np.ndarray  # (8,) raw logits from orientation head
    keypoints: List[Keypoint] # all keypoints, always the full list
    image_wh: Tuple[int, int] # (width, height) of the source image

    def visible_keypoints(self, conf_thresh: Optional[float] = None) -> List[Keypoint]:
        """Return keypoints whose score >= conf_thresh (defaults to 0.0 = all)."""
        thr = conf_thresh if conf_thresh is not None else 0.0
        return [kp for kp in self.keypoints if kp.score >= thr]

    def to_dict(self) -> dict:
        """JSON-serialisable dict — handy for logging / message queues."""
        return {
            'orientation':        self.orientation,
            'orientation_idx':    self.orientation_idx,
            'orientation_scores': self.orientation_scores.tolist(),
            'image_wh':           self.image_wh,
            'keypoints': [
                {'id': kp.id, 'label': kp.label,
                 'x': round(kp.x, 1), 'y': round(kp.y, 1),
                 'score': round(kp.score, 4)}
                for kp in self.keypoints
            ],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class VehicleLandmarkDetector:
    """
    Wraps an ONNX stage-2 KeyPointModel for use in any surveillance pipeline.

    Thread-safety: onnxruntime InferenceSession.run() is thread-safe; multiple
    threads may call detect() on the same instance concurrently.

    Parameters
    ----------
    model_path    : path to stage2.onnx  (produced by export_onnx.py)
    mean_std_path : path to stage2_mean_std.npz  (produced by export_onnx.py)
    device        : 'cpu' or 'cuda'
    conf_thresh   : default confidence threshold used by draw(); detect()
                    always returns the full keypoint list regardless
    """

    def __init__(
        self,
        model_path: str,
        mean_std_path: str,
        device: str = 'cpu',
        conf_thresh: float = 0.0,
    ) -> None:
        import onnxruntime as ort

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'ONNX model not found: {model_path}')
        if not os.path.isfile(mean_std_path):
            raise FileNotFoundError(f'mean/std file not found: {mean_std_path}')

        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._active_provider = self._session.get_providers()[0]

        ms = np.load(mean_std_path)
        self._mean = ms['mean'].astype(np.float64)  # (3,) — RGB, range 0-255
        self._std  = ms['std'].astype(np.float64)

        self.conf_thresh = conf_thresh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: Union[np.ndarray, str]) -> LandmarkDetection:
        """
        Run inference on a single image.

        Parameters
        ----------
        image : BGR numpy array (H, W, 3) uint8  **or**  path string to an image file.

        Returns
        -------
        LandmarkDetection with all keypoints (unfiltered).
        """
        bgr = self._load_bgr(image)
        orig_h, orig_w = bgr.shape[:2]

        t224, t56 = self._preprocess(bgr)
        _coarse, fine_kp, ori_logits = self._session.run(None, {
            'image_224': t224,
            'image_56':  t56,
        })

        return self._decode(fine_kp, ori_logits, orig_h, orig_w)

    def draw(
        self,
        image: Union[np.ndarray, str],
        result: LandmarkDetection,
        conf_thresh: Optional[float] = None,
        radius: int = 8,
        show_legend: bool = True,
    ) -> np.ndarray:
        """
        Render keypoints and orientation onto a copy of *image*.

        Parameters
        ----------
        image       : original BGR image or path
        result      : LandmarkDetection returned by detect()
        conf_thresh : override the instance default for this call
        radius      : keypoint circle radius in pixels
        show_legend : append a legend panel on the right

        Returns
        -------
        BGR numpy array (annotated copy — source image is not modified)
        """
        thr = conf_thresh if conf_thresh is not None else self.conf_thresh
        bgr = self._load_bgr(image).copy()
        orig_h, orig_w = bgr.shape[:2]

        visible = result.visible_keypoints(thr)

        # --- keypoint dots + index/score labels ---
        for kp in visible:
            cx, cy = int(round(kp.x)), int(round(kp.y))
            color   = _PALETTE_BGR[kp.id % len(_PALETTE_BGR)]
            cv2.circle(bgr, (cx, cy), radius, color, -1)
            cv2.circle(bgr, (cx, cy), radius, (255, 255, 255), 1)
            cv2.putText(bgr, f'#{kp.id} {kp.score:.3f}',
                        (cx + radius + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # --- orientation banner ---
        banner_h = 28
        banner   = np.zeros((banner_h, orig_w, 3), dtype=np.uint8)
        cv2.putText(banner,
                    f'Orientation: {result.orientation}',
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 180), 2, cv2.LINE_AA)

        canvas = np.vstack([banner, bgr])

        # --- legend panel ---
        if show_legend and visible:
            legend_w  = 240
            row_h     = 16
            legend_h  = max(canvas.shape[0], len(visible) * row_h + 10)
            legend    = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
            for row, kp in enumerate(visible):
                y_pos = row * row_h + row_h
                color = _PALETTE_BGR[kp.id % len(_PALETTE_BGR)]
                cv2.rectangle(legend, (4, y_pos - 9), (14, y_pos + 3), color, -1)
                cv2.putText(legend, f'#{kp.id} {kp.label}',
                            (18, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (200, 200, 200), 1, cv2.LINE_AA)
            pad = legend_h - canvas.shape[0]
            if pad > 0:
                canvas = np.vstack([canvas,
                                    np.zeros((pad, canvas.shape[1], 3), dtype=np.uint8)])
            canvas = np.hstack([canvas, legend])

        return canvas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_bgr(image: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise FileNotFoundError(f'Cannot read image: {image}')
            return bgr
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected BGR numpy array or file path, got {type(image)}')
        return image

    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """BGR uint8 -> two float32 NCHW tensors for the ONNX session."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
        for c in range(3):
            rgb[:, :, c] = (rgb[:, :, c] - self._mean[c]) / self._std[c]
        img_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_56  = cv2.resize(rgb, (56,  56),  interpolation=cv2.INTER_LINEAR)
        t224 = img_224.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        t56  = img_56.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return t224, t56

    @staticmethod
    def _decode(
        fine_kp: np.ndarray,    # (1, C, 56, 56)
        ori_logits: np.ndarray, # (1, 8)
        orig_h: int,
        orig_w: int,
    ) -> LandmarkDetection:
        # Orientation
        scores      = ori_logits[0]
        pred_cls    = int(np.argmax(scores))
        ori_label   = ORIENTATION_LABELS[pred_cls]

        # Keypoints
        hm      = fine_kp[0]  # (C, 56, 56)
        kpoints = []
        for c in range(hm.shape[0]):
            flat  = hm[c].ravel()
            idx   = int(flat.argmax())
            x56   = idx % 56
            y56   = idx // 56
            label = KEYPOINT_LABELS[c] if c < len(KEYPOINT_LABELS) else f'kp_{c}'
            kpoints.append(Keypoint(
                id    = c,
                label = label,
                x     = x56 * orig_w / 56,
                y     = y56 * orig_h / 56,
                score = float(flat[idx]),
            ))

        return LandmarkDetection(
            orientation        = ori_label,
            orientation_idx    = pred_cls,
            orientation_scores = scores,
            keypoints          = kpoints,
            image_wh           = (orig_w, orig_h),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f'VehicleLandmarkDetector('
                f'provider={self._active_provider}, '
                f'conf_thresh={self.conf_thresh})')
