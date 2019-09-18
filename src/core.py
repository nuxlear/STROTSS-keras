import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .utils import *


def laplacian_pyramid(imgs, steps: int=1):
    if len(imgs.shape) != 4:
        raise AssertionError('the image for calculate Laplacian pyramid must be 4, received {}'.format(imgs.ndim))
    
    hf, wf = 2**(steps - 1), 2**(steps - 1)
    h, w = imgs.shape[1] // hf, imgs.shape[2] // wf
    
    results = []
    for img in imgs:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        tmp = cv2.resize(img, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_LINEAR)
        tmp = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)

        n = cv2.subtract(img, tmp)
        results.append(n)
        
    return np.array(results)


def run(style_img, content_img, content_weight=1, max_scale=5):
    
    small_sz = 64

    content_img_big = scale_max(content_img, 512)[np.newaxis, :]

    for scale in range(1, max_scale):
        pass


if __name__ == '__main__':
    img = imageio.imread('images/butterfly.jpg')
    
    lap = laplacian_pyramid(img[np.newaxis, :])

    plt.imshow(lap[0])
    plt.show()
