from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps
import pymupdf

from .models import QualityResult


MAX_FILE_BYTES = 12_000_000
MAX_PIXELS = 30_000_000


def _render_pdf_pages(blob: bytes) -> list[Image.Image]:
    """Render every PDF page at a resolution suitable for document OCR."""
    try:
        document = pymupdf.open(stream=blob, filetype="pdf")
        if document.needs_pass:
            raise ValueError("PDF защищён паролем")
        if document.page_count == 0:
            raise ValueError("PDF не содержит страниц")
        images = []
        for page in document:
            # 200 dpi keeps passport glyphs legible without exceeding the image cap
            # for normal A4/Letter scans.
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(200 / 72, 200 / 72), alpha=False)
            if pixmap.width * pixmap.height > MAX_PIXELS:
                raise ValueError("Страница PDF превышает лимит 30 Мп")
            images.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
        return images
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Не удалось прочитать PDF. Загрузите корректный PDF без пароля.") from exc
    finally:
        if "document" in locals():
            document.close()


def decode_image(blob: bytes) -> np.ndarray:
    return decode_document_pages(blob)[0]


def decode_document_pages(blob: bytes) -> list[np.ndarray]:
    """Decode an image or every page of a PDF into BGR images."""
    if len(blob) > MAX_FILE_BYTES: raise ValueError("Файл больше лимита 12 МБ")
    try:
        if blob.lstrip().startswith(b"%PDF-"):
            pages = _render_pdf_pages(blob)
        else:
            pages = [ImageOps.exif_transpose(Image.open(__import__("io").BytesIO(blob))).convert("RGB")]
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Не удалось декодировать файл. Используйте JPEG, PNG или PDF.") from exc
    decoded = []
    for page in pages:
        if page.width * page.height > MAX_PIXELS: raise ValueError("Изображение превышает лимит 30 Мп")
        decoded.append(cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR))
    return decoded


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
    # Passport data pages are commonly portrait-oriented while their text and MRZ
    # are already horizontal. Rotating solely from the aspect ratio destroys OCR.
    # EXIF orientation has already been applied in decode_image().
    h, w = image.shape[:2]
    max_width = 2200
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif image.shape[1] < 1400:
        # Small source thumbnails lose MRZ glyph detail. Cubic upscaling gives
        # the local recognizer enough character pixels without inventing text.
        scale = 1400 / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


def verification_variant(image: np.ndarray) -> np.ndarray:
    """Build an independent, contrast-normalized page for the second OCR pass."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, 5, 30, 30)


def mrz_variants(image: np.ndarray) -> list[np.ndarray]:
    h, _ = image.shape[:2]
    crop = image[int(h * .65):h, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 13)
    return [crop, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)]


def name_region_variant(image: np.ndarray) -> np.ndarray:
    """Upscale the visual identity band where short names are often missed."""
    height, width = image.shape[:2]
    crop = image[int(height * .24):int(height * .62), :]
    target_width = min(2600, max(width, 2200))
    scale = target_width / max(width, 1)
    if scale > 1.02:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop
