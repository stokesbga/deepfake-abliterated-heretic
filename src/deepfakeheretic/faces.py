from pathlib import Path

import cv2
import numpy as np


class FaceMasker:
    """Upstream DWPose conditioning on CPU, selecting the largest visible face."""

    def __init__(self, models: Path):
        from ._vendor.dwpose.wholebody import Wholebody

        self.pose = Wholebody(
            str(models / "yolox_l.onnx"), str(models / "dw-ll_ucoco_384.onnx"), device="cpu"
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        # Match the upstream annotation resolution and confidence threshold.
        scale = 1024 / min(height, width)
        resized = cv2.resize(
            image,
            (round(width * scale / 64) * 64, round(height * scale / 64) * 64),
            interpolation=cv2.INTER_LANCZOS4,
        )
        candidate, scores, _ = self.pose(resized)
        faces = candidate[:, 24:92].copy()
        faces[..., 0] *= width / resized.shape[1]
        faces[..., 1] *= height / resized.shape[0]
        valid = scores[:, 24:92] >= 0.3
        mask = np.zeros((height, width), dtype=np.uint8)
        hulls = [
            cv2.convexHull(face[ok].astype(np.int32))
            for face, ok in zip(faces, valid)
            if ok.sum() >= 8
        ]
        if hulls:
            hull = max(hulls, key=cv2.contourArea)
            if cv2.contourArea(hull) >= 16:
                cv2.fillConvexPoly(mask, hull, 255)
                mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        return mask


def composite(original: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restore original pixels outside a softly feathered face region."""
    height, width = original.shape[:2]
    generated = cv2.resize(generated, (width, height), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    sigma = max(1.0, min(height, width) / 160)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), sigma)[..., None]
    return np.clip(np.rint(generated * alpha + original * (1 - alpha)), 0, 255).astype(np.uint8)
