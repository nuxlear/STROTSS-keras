import imageio
import cv2
import numpy as np
from keras.layers import *
from keras.models import Model
import keras.backend as K
import tensorflow as tf


# def extract_style_from_image(img, n_samples, scale, inner):
        
#     from model import VGG16_pt
    
#     z_img = load_image(img, max_side=scale, force_scale=True)
#     x = Input(shape=z_img.shape[1:])
#     vgg = VGG16_pt(z_img.shape[1:], inference_type='cat', n_samples=n_samples)
#     extractor = Model(x, vgg(x), name='extractor_vgg_pt')
    
#     zs = []
#     for i in range(inner):
#         zs.append(extractor.predict(z_img))
#     z = np.concatenate(zs, axis=2)
    
#     return z, z_img


def load_image(image, max_side=1000, force_scale=False, normalize=True):

    x = image
    if isinstance(image, str):
        x = imageio.imread(image).astype('float32')
    assert isinstance(x, np.ndarray)
    
    if normalize:
        x = normalize_image(x, -0.5, 0.5)

    if len(x.shape) < 3:
        x = np.stack([x, x, x], 2)
    if x.shape[2] > 3:
        x = x[..., :3]

    if len(x.shape) == 3:
        x = x[np.newaxis, :]

    w, h = x.shape[-3:-1]
    if 0 < max_side < max(w, h) or force_scale:
        x = scale_max(x, max_side)

    return x


def normalize_image(image, low=0, high=1):
    assert low < high

    d = high - low
    return (image / 255.) * d + low


def resize(imgs, size=None, ratio=None, is_tensor=False):
    if len(imgs.shape) != 4:
        raise AssertionError('the image for resize must be 4, received {}'.format(imgs.ndim))

    if size is None and ratio is None:
        raise ValueError('Both of size and ratio must not be None. ')

    if ratio is not None:
        h, w = int(imgs.shape[1] * ratio), int(imgs.shape[2] * ratio)
    if size is not None:
        h, w = size

    if is_tensor:
        return tf.image.resize(imgs, (h, w))
    
    resized = []
    for img in imgs:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        resized.append(img)

    return np.array(resized)


def scale_max(x, max_side=1000., is_tensor=False):
    w, h = x.shape[-3:-1]
    factor = max_side / max(w, h)
    x = resize(x, ratio=factor, is_tensor=is_tensor)

    return x


def rgb_to_yuv(x):
    # I cannot understand why the value below is differ from general rgb-to-yuv formula, 
    # but it works, though. 
    c = K.constant([[0.577350,0.577350,0.577350],
                    [-0.577350,0.788675,-0.211325],
                    [-0.577350,-0.211325,0.788675]])
    
    return K.dot(x, K.transpose(c))


if __name__ == '__main__':
    img = load_image('images/butterfly.jpg')
    long_side = 512

    x = Input(shape=img.shape[1:])
    model = Model(x, Lambda(lambda x: rgb_to_yuv(x))(x))

    yuv = model.predict(img)
    print(yuv.shape)
    print(yuv.min(), yuv.max())

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(np.mean(img[0] * .5, axis=-1), cmap='gray')
    axes[1].imshow(yuv[0, ..., 0] * .5, cmap='gray')
    plt.show()
    
    # z, z_img = extract_style_from_image(img, 1000, long_side, 5)
    # print('z: {}'.format(z.shape))
    # print('z_img: {}'.format(z_img.shape))
