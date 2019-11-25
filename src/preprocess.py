import keras.backend as K
import tensorflow as tf

from src.utils import *
from src.model import *


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
