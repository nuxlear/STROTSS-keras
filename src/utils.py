import imageio
import cv2
import numpy as np


def load_image(image, max_side=1000, force_scale=False, normalize=True):

    x = image
    if isinstance(image, str):
        x = imageio.imread(image).astype('float32')
    assert isinstance(x, np.ndarray)
    
    if normalize:
        x = normalize_image(x, -1, 1)

    if len(x.shape) < 3:
        x = np.stack([x, x, x], 2)
    if x.shape[2] > 3:
        x = x[..., :3]

    w, h = x.shape[-3:-1]
    if 0 < max_side < max(w, h) or force_scale:
        x = scale_max(x, max_side)

    return x


def normalize_image(image, low=0, high=1):
    assert low < high

    d = high - low
    return (image / 255.) * d + low


def scale_max(x, max_side=1000.):
    w, h = x.shape[-3:-1]
    factor = max_side / max(w, h)
    cv2.resize(x, (int(factor * w), int(factor * h)), interpolation='bilinear')

    return x
