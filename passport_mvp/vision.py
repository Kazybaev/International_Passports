from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

from .models import QualityResult


def decode_image(blob: bytes) -> np.ndarray:
    if len(blob) > 12_000_000: raise ValueError("Файл больше лимита 12 МБ")
    try:
        pil = ImageOps.exif_transpose(Image.open(__import__("io").BytesIO(blob))).convert("RGB")
    except Exception as exc:
        raise ValueError("Не удалось декодировать изображение. Используйте JPEG или PNG.") from exc
    if pil.width * pil.height > 30_000_000: raise ValueError("Изображение превышает лимит 30 Мп")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def quality(image: np.ndarray) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    glare = float(np.mean(gray > 248))
    h, w = gray.shape
    reasons = []
    if min(h, w) < 700: reasons.append("LOW_RESOLUTION")
    if blur < 75: reasons.append("IMAGE_BLUR")
    if brightness < 45: reasons.append("TOO_DARK")
    if brightness > 225: reasons.append("OVEREXPOSED")
    if glare > 0.09: reasons.append("EXCESSIVE_GLARE")
    return QualityResult(blur, brightness, glare, f"{w}×{h}", "retry" if reasons else "ok", reasons)


def normalize(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w < h: image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    max_width = 2200
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def mrz_variants(image: np.ndarray) -> list[np.ndarray]:
    h, _ = image.shape[:2]
    crop = image[int(h * .62):h, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 13)
    return [crop, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)]

