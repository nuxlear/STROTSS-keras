import sys
sys.path.append('.')

from keras.optimizers import RMSprop
import keras.backend as K

import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from src.model import *
from src.loss import *
from src.preprocess import *


def preprocess_style_image(img, n_samples=1024, scale=512, inner=1):
    z_img = load_image(img, max_side=scale, force_scale=True)
    x = Input(shape=z_img.shape[1:])
    vgg = VGG16_pt(z_img.shape[1:], inference_type='cat', n_samples=n_samples)
    model = Model(x, vgg(x), name='vgg_pt_cat')

    zs = []
    for i in range(inner):
        zs.append(model.predict(z_img))
    z = np.concatenate(zs, axis=2)

    return z, z_img

def preprocess_content_image(img, n_samples=1024):
    x = Input(shape=img.shape[1:])
    vgg = VGG16_pt(img.shape[1:], inference_type='normal', n_samples=n_samples)
    model = Model(x, vgg(x), name='vgg_pt_normal')

    return model.predict(img)


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


def style_transfer(base_img, style_img, content_img, long_side, lr=1e-3, content_weight=1.):

    ## Definitions

    n_iter = 201
    n_eval_step = 100
    
    optimizer = RMSprop(lr=lr)

    ## Preparation

    st_model = StyleTransfer(base_img, style_img, content_img, long_side)
    st_model.compile(optimizer=optimizer)

    ## Training

    for i in range(n_iter):
        st_model.train_on_batch()
        if i % n_eval_step == 0:
            losses = st_model.test_on_batch()
            print(losses)

    return K.eval(st_model.x_img), st_model.test_on_batch()


def run(style_img, content_img, content_weight=16, max_scale=5):
    
    small_sz = 64
    lrs = [2e-3] * (max_scale - 1) + [1e-3]

    content_img_big = scale_max(content_img, 512)

    for scale in range(1, max_scale + 1):

        long_side = small_sz * (2**(scale - 1))
        lr = lrs[scale]

        style = scale_max(style_img, long_side)
        content = scale_max(content_img, long_side)
        style_mean = np.mean(np.mean(
            scale_max(style_img, long_side), 1, keepdims=True), 2, keepdims=True)

        lap = laplacian_pyramid(content)
        nz = np.random.normal(0., 0.1, size=lap.shape)

        # canvas = resize(np.clip(lap, -0.5, 0.5), size=content_img_big.shape[1:3])
        
        if scale == 1:
            # canvas = resize(content, ratio=1/2)
            base = style_mean + lap
        else:
            base = resize(base, size=content.shape[1:3])
            if scale < max_scale - 1:
                base += lap

        base, loss = style_transfer(base, style, content, long_side=long_side,
                                    lr=lr, content_weight=content_weight)

        base = np.clip(base, -0.5, 0.5)

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(base[0] + .5)
        ax[1].imshow(content[0] + .5)
        plt.show()

        # canvas = resize(np.clip(style, -0.5, 0.5), size=(content.shape[1]//2, content.shape[2]//2))

        content_weight /= 2
    
    # return loss, style


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    style = load_image('../images/sketch.jpg')
    content = load_image('../images/crayon.jpg')

    run(style, content)
    # lap = laplacian_pyramid(img[np.newaxis, :])

    # plt.imshow(lap[0])
    # plt.show()
