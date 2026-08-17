import numpy as np
import pymupdf

from passport_mvp.vision import decode_document_pages, decode_image, name_region_variant, normalize, verification_variant


def test_normalize_preserves_portrait_orientation_and_upscales():
    image = np.zeros((512, 307, 3), dtype=np.uint8)
    result = normalize(image)
    assert result.shape[0] > result.shape[1]
    assert result.shape[1] == 1400


def test_normalize_caps_large_image_width():
    image = np.zeros((1800, 3000, 3), dtype=np.uint8)
    result = normalize(image)
    assert result.shape[1] == 2200


def test_name_region_variant_crops_and_upscales_identity_band():
    image = np.zeros((1000, 1400, 3), dtype=np.uint8)

    result = name_region_variant(image)

    assert result.shape[1] == 2200
    assert 580 < result.shape[0] < 620


def test_verification_variant_preserves_page_geometry():
    image = np.full((200, 300, 3), 120, dtype=np.uint8)

    result = verification_variant(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_decode_image_renders_first_pdf_page():
    document = pymupdf.open()
    first = document.new_page(width=144, height=216)
    first.draw_rect(pymupdf.Rect(0, 0, 144, 216), color=None, fill=(1, 0, 0))
    document.new_page(width=216, height=144)
    blob = document.tobytes()
    document.close()

    result = decode_image(blob)

    assert result.shape[:2] == (600, 400)
    assert result[300, 200, 2] > 240


def test_decode_document_pages_renders_every_pdf_page():
    document = pymupdf.open()
    document.new_page(width=144, height=216)
    document.new_page(width=216, height=144)
    pages = decode_document_pages(document.tobytes())
    document.close()

    assert [page.shape[:2] for page in pages] == [(600, 400), (400, 600)]


def test_decode_document_pages_has_no_page_count_limit():
    document = pymupdf.open()
    for _ in range(21):
        document.new_page(width=72, height=72)

    pages = decode_document_pages(document.tobytes())
    document.close()

    assert len(pages) == 21
