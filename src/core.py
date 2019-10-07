import sys
sys.path.append('.')

from keras.optimizers import RMSprop

import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from src.model import *
from src.loss import *


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


def style_transfer(style_img, content_img, lr=1e-3, content_weight=1.):

    ## Definitions

    MAX_ITER = 250

    cnn = vgg16_pt()

    phi = lambda x: cnn.predict(x)
    # phi2 = lambda x, y, z: 

    optimizer = RMSprop(lr=lr)
    loss = REMD_loss

    z_c = phi(content_img)

    z_s_all = []

    # requires style transfer model, wrapper of VGG16_pt model

    ## Preparation


    ## Training

    for i in range(MAX_ITER):
        pass


def run(style_img, content_img, content_weight=16, max_scale=5):
    
    small_sz = 64
    lrs = [2e-3] * (max_scale - 1) + [1e-3]

    content_img_big = scale_max(content_img, 512)

    for scale in range(1, max_scale):

        long_side = small_sz * (2**(scale - 1))
        lr = lrs[scale]

        content = scale_max(content_img, long_side)
        style_mean = np.mean(np.mean(
            scale_max(style_img, long_side), 1, keepdims=True), 2, keepdims=True)

        lap = laplacian_pyramid(content)
        nz = np.random.normal(0., 0.1, size=lap.shape)

        # canvas = resize(np.clip(lap, -0.5, 0.5), size=content_img_big.shape[1:3])
        
        if scale == 1:
            # canvas = resize(content, ratio=1/2)
            style = style_mean + lap
        else:
            style = resize(style, size=content.shape[1:3])
            if scale < max_scale - 1:
                style += lap

        # style, loss = style_transfer(style, content, content_weight=content_weight)

        style = np.clip(style, -0.5, 0.5)

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(style[0] + .5)
        ax[1].imshow(content[0] + .5)
        plt.show()

        # canvas = resize(np.clip(style, -0.5, 0.5), size=(content.shape[1]//2, content.shape[2]//2))

        content_weight /= 2
    
    # return loss, style


if __name__ == '__main__':
    style = load_image('images/crayon.jpg')
    content = load_image('images/butterfly.jpg')
    
    run(style, content)
    # lap = laplacian_pyramid(img[np.newaxis, :])

    # plt.imshow(lap[0])
    # plt.show()
