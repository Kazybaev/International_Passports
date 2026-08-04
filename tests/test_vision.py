import numpy as np

from passport_mvp.vision import normalize


def test_normalize_preserves_portrait_orientation_and_upscales():
    image = np.zeros((512, 307, 3), dtype=np.uint8)
    result = normalize(image)
    assert result.shape[0] > result.shape[1]
    assert result.shape[1] == 1400


def test_normalize_caps_large_image_width():
    image = np.zeros((1800, 3000, 3), dtype=np.uint8)
    result = normalize(image)
    assert result.shape[1] == 2200
