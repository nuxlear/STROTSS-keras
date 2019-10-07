import sys
sys.path.append('.')

from keras.layers import *
from keras.models import Model
from keras.applications import VGG16
import tensorflow as tf

import numpy as np

from src.utils import *


# def vgg16_pt(input_shape):
#     vgg = VGG16(weights='imagenet')
#     extracted_layers = [1, 2, 4, 5, 7, 8, 9, 13, 17]

#     x = Input(shape=input_shape)

#     outputs = []
#     for l in extracted_layers:
#         out = x
#         for i in range(l):
#             out = vgg.layers[i](out)
#         outputs.append(out)
    
#     # codes corresponding with `forward_cat`


# def dec_lap_pyr(imgs, steps: int=1):
#     if len(imgs.shape) != 4:
#         raise AssertionError('the image for calculate Laplacian pyramid must be 4, received {}'.format(imgs.ndim))
    
#     results = []
#     cur = imgs
#     for i in range(steps):
#         h, w = imgs.shape[-3:-1]
#         small = tf.image.resize(imgs, (max(h//2, 1), max(w//2, 1)))
#         back = tf.image.resize(small, (h, w))

#         results.append(cur - back)
#         cur = small
    
#     results.append(cur)
#     return results

# def syn_lap_pyr(pyr):
#     pass


class LaplacianPyramid(Model):

    def __init__(self, levels=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels

    def call(self, inputs):
        results = []
        cur = inputs
        for i in range(self.levels):
            h, w = cur.shape[1:3]
            small = tf.image.resize(cur, (max(h//2, 1), max(w//2, 1)))
            back = tf.image.resize(small, (h, w))

            results.append(cur - back)
            cur = small
        
        results.append(cur)
        return results

    def compute_output_shape(self, input_shape):
        shapes = []
        b, h, w, c = input_shape
        for i in range(self.levels):
            shapes.append((b, h, w, c))
            h = max(h // 2, 1)
            w = max(w // 2, 1)
        
        shapes.append((b, h, w, c))
        return shapes


class InverseLaplacianPyramid(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        
        cur, pyrs = inputs[-1], inputs[:-1]
        for pyr in pyrs[::-1]:
            h, w = pyr.shape[1:3]
            cur = pyr + tf.image.resize(cur, (h, w))

        return cur

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VGG16_pt(Model):

    def __init__(self, input_shape, inference_type=None, n_samples=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        self.extracted_layers = [1, 2, 4, 5, 7, 8, 9, 13, 17]

        if inference_type is None:
            inference_type = 'cat'
        assert inference_type in ['normal', 'cat', 'diff']
        self.inference_type = inference_type
        
        if inference_type == 'cat':
            self.n_samples = n_samples

        self.vgg.trainable=False

    def build(self, input_shape):
        x = Input(shape=input_shape[1:])

        outputs = []
        for l in self.extracted_layers:
            out = x
            for i in range(l):
                out = self.vgg.layers[i](out)
            outputs.append(out)
        
        self.models = [Model(x, out, name='feat_{}'.format(i)) for i, out in enumerate(outputs)]
        super().build(input_shape)

    def call(self, inputs):
        if self.inference_type == 'normal':
            return self._call_normal(inputs)
        if self.inference_type == 'cat':
            return self._call_cat(inputs)
        raise ValueError('invalid inference type: {}'.format(self.inference_type))

    def _call_normal(self, inputs):
        return [model(inputs) for model in self.models]

    def _call_cat(self, inputs):
        outputs = self._call_normal(inputs)

        xx, xy = np.meshgrid(np.array(range(inputs.shape[1])), np.array(range(inputs.shape[2])))
        xx = np.expand_dims(xx.flatten(), 1)
        xy = np.expand_dims(xy.flatten(), 1)
        xc = np.concatenate([xx, xy], axis=1)

        pass

    def compute_output_shape(self, input_shape):
        return [model.output_shape for model in self.models]


class StyleTransfer(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        pass


if __name__ == '__main__':

    img = load_image('images/butterfly.jpg')
    print(img.shape)

    # test vgg_pt
    x = Input(shape=img.shape[1:])
    vgg = VGG16_pt(input_shape=img.shape[1:], inference_type='normal')

    model = Model(x, vgg(x), name='extractor')
    model.summary()

    pred = model.predict(img)
    for p in pred:
        print(p.shape)

    # test lap_pyr
    lap = LaplacianPyramid(levels=5)
    l_model = Model(x, lap(x), name='lap_pyr')
    l_model.summary()

    hc = l_model.predict(img)
    for h in hc:
        print(h.shape)

    # test inv_l_p
    xs = []
    for shape in l_model.output_shape:
        xs.append(Input(shape=shape[1:]))
    inv = InverseLaplacianPyramid()
    i_model = Model(xs, inv(xs), name='inv_l_p')
    i_model.summary()

    res = i_model.predict(hc)
    print(res.shape)