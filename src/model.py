import sys
sys.path.append('.')

from keras.layers import *
from keras.models import Model
from keras.applications import VGG16
import keras.backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

from src.utils import *
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


class LaplacianPyramid(Layer):

    def __init__(self, levels=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels

    def call(self, inputs, mask=None):
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


class InverseLaplacianPyramid(Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        
        cur, pyrs = inputs[-1], inputs[:-1]
        for pyr in pyrs[::-1]:
            h, w = pyr.shape[1:3]
            cur = pyr + tf.image.resize(cur, (h, w))

        return cur

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VGG16_pt(Layer):

    def __init__(self, inputs, inference_type=None, n_samples=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(inputs, tuple):
            self.vgg = VGG16(weights='imagenet', include_top=False, input_shape=inputs)
        elif not K.is_variable(inputs):
            self.vgg = VGG16(weights='imagenet', include_top=False, input_shape=K.int_shape(inputs)[1:])
        else:
            self.vgg = VGG16(weights='imagenet', include_top=False,
                             input_tensor=inputs, input_shape=K.int_shape(inputs)[1:])
        self.extracted_layers = [1, 2, 4, 5, 7, 8, 9, 13, 17]

        if inference_type is None:
            inference_type = 'cat'
        assert inference_type in ['normal', 'cat', 'diff']
        self.inference_type = inference_type
        
        if inference_type == 'cat':
            self.n_samples = n_samples

        self.vgg.trainable = False

    def build(self, input_shape):
        x = Input(shape=input_shape[1:])

        outputs = [x]
        for l in self.extracted_layers:
            out = x
            for i in range(1, l+1):
                out = self.vgg.layers[i](out)
            outputs.append(out)
        
        self.models = [Model(x, out, name='feat_{}'.format(i)) for i, out in enumerate(outputs)]
        super().build(input_shape)

    def call(self, inputs, mask=None):
        if self.inference_type == 'normal':
            return self._call_normal(inputs)
        if self.inference_type == 'cat':
            return self._call_cat(inputs, mask)
        raise ValueError('invalid inference type: {}'.format(self.inference_type))

    def _call_normal(self, inputs):
        return [model(inputs) for model in self.models]

    def _call_cat(self, inputs, mask=None):
        outputs = self._call_normal(inputs)

        xx, xy = tf.meshgrid(K.arange(inputs.shape[1]), K.arange(inputs.shape[2]))
        xx = K.expand_dims(K.flatten(xx), 1)
        xy = K.expand_dims(K.flatten(xy), 1)
        xc = K.concatenate([xx, xy], axis=1)

        if mask is not None:
            xc = xc[mask, :]

        n_samples = min(self.n_samples, xc.shape[0])

        xx = xc[:n_samples, 0]
        yy = xc[:n_samples, 1]

        ## Need to understand of PyTorch tensor shaping ##

        ls = []
        for i, out in enumerate(outputs):
            b, w, h, c = out.shape
            x = out

            if i > 0 and out.shape[1] < outputs[i-1].shape[1]:
                xx = K.cast(xx, 'float32') / 2.
                yy = K.cast(yy, 'float32') / 2.

            xx = K.cast(K.clip(xx, 0, out.shape[1]-1), 'int32')
            yy = K.cast(K.clip(yy, 0, out.shape[2]-1), 'int32')

            idx = xx * h + yy
            x = tf.gather(K.reshape(x, (-1, w * h, c)), idx, axis=1)
            x = K.expand_dims(x, axis=2)

            ls.append(x)    # NOTICE: the original code do clone() and detach()
        
        out = K.concatenate(ls, axis=-1)    # NOTICE: the original code do contiguous()
        return out

    def compute_output_shape(self, input_shape):
        b = input_shape[0]
        if self.inference_type == 'normal':
            return [(b,) + model.get_output_at(-1).shape[1:] for model in self.models]
        if self.inference_type == 'cat':
            ch = sum([model.output_shape[-1] for model in self.models])
            n = min(input_shape[1] * input_shape[2], self.n_samples)
            return b, n, 1, ch


class StyleTransfer(Model):

    def __init__(self, base_img, style_img, content_img, scale, n_samples=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pyr = self.__get_laplacian_pyramid_model(base_img)
        self.x_imgs = [K.variable(t) for t in self.pyr.predict(base_img)]

        self.inv_pyr = self.__get_inv_laplacian_pyramid_model(self.x_imgs)
        # self.x_img = K.variable(base_img)

        self.x_img = self.inv_pyr(self.x_imgs)
        
        self.vgg = VGG16_pt(self.x_img, inference_type='normal', n_samples=n_samples)

        self.objective = objective_function

        # if not os.path.isfile('tmp_z_s.pkl'):
        #     self.z_s = preprocess_style_image(style_img, n_samples=n_samples, scale=scale, inner=1)
        #     with open('tmp_z_s.pkl', 'wb') as f:
        #         pickle.dump(self.z_s, f)
        # if not os.path.isfile('tmp_z_c.pkl'):
        #     self.z_c = preprocess_content_image(content_img, n_samples=n_samples)
        #     with open('tmp_z_c.pkl', 'wb') as f:
        #             pickle.dump(self.z_c, f)
        #
        # with open('tmp_z_s.pkl', 'rb') as f:
        #     self.z_s = pickle.load(f)
        # with open('tmp_z_c.pkl', 'rb') as f:
        #     self.z_c = pickle.load(f)

        print('Preprocessing...', flush=True)
        self.z_s = preprocess_style_image(style_img, n_samples=n_samples, scale=scale, inner=1)
        print('Preprocess [style] Finished', flush=True)
        self.z_c = preprocess_content_image(content_img, n_samples=n_samples)
        print('Preprocess [content] Finished', flush=True)
        print('Preprocess Finished: {}'.format(base_img.shape), flush=True)

        self.train_function = None
        self.test_function = None

    def __get_laplacian_pyramid_model(self, img):
        pyr = LaplacianPyramid(5)
        x = Input(shape=img.shape[1:])
        return Model(x, pyr(x), name='lap_pyr')

    def __get_inv_laplacian_pyramid_model(self, pyrs):
        inv = InverseLaplacianPyramid()
        xs = [Input(shape=x.shape[1:]) for x in pyrs]
        return Model(xs, inv(xs), name='inv_lap_pyr')

    def __init_train_function(self):
        z_x = self.vgg(self.x_img)
        loss = objective_function(z_x, self.z_s, self.z_c)
        updates = self.__update_image(self.x_imgs, loss)

        self.train_function = K.function([],
                                         [loss],
                                         updates=updates)

    def __init_test_function(self):
        z_x = self.vgg(self.x_img)
        loss = objective_function(z_x, self.z_s, self.z_c)

        self.test_function = K.function([],
                                        [loss])

    def call(self, inputs, mask=None):
        return self.x_img

    def __update_image(self, x, loss):
        if not hasattr(self, 'optimizer'):
            raise RuntimeError('You must be compile your model before do style-transferring.')

        # self.x_img._keras_shape = self.x_img._shape
        for t in x:
            t._keras_shape = t._shape
        updates = self.optimizer.get_updates(loss, x)
        return updates

    def train_on_batch(self, x=None, y=None,
                       sample_weight=None,
                       class_weight=None,
                       reset_metrics=True):

        # randomly choose region and update

        if self.train_function is None:
            self.__init_train_function()

        return self.train_function([])

    def test_on_batch(self, x=None, y=None,
                      sample_weight=None,
                      reset_metrics=True):

        if self.test_function is None:
            self.__init_test_function()

        return self.test_function([])

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + tuple(self.x_img.shape)[1:]


if __name__ == '__main__':

    # tf.compat.v1.disable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    c_img = load_image('images/butterfly.jpg')
    s_img = load_image('images/crayon.jpg')
    print(c_img.shape)
    print(s_img.shape)

    # # test vgg_pt
    # x = Input(shape=img.shape[1:])
    # vgg = VGG16_pt(input_shape=img.shape[1:], inference_type='cat')

    # model = Model(x, vgg(x), name='extractor')
    # model.summary()

    # pred = model.predict(img)
    # if isinstance(pred, list):
    #     for p in pred:
    #         print(p.shape)
    # else:
    #     print(pred.shape)

    # # test lap_pyr
    # lap = LaplacianPyramid(levels=5)
    # l_model = Model(x, lap(x), name='lap_pyr')
    # l_model.summary()

    # hc = l_model.predict(img)
    # for h in hc:
    #     print(h.shape, h.min(), h.max())
    #     plt.imshow(np.clip(h[0] + .5, 0., 1.))
    #     plt.show()

    # # test inv_l_p
    # xs = []
    # for shape in l_model.output_shape:
    #     xs.append(Input(shape=shape[1:]))
    # inv = InverseLaplacianPyramid()
    # i_model = Model(xs, inv(xs), name='inv_l_p')
    # i_model.summary()

    # res = i_model.predict(hc)
    # print(res.shape, res.min(), res.max())
    # plt.imshow(np.clip(res[0] + .5, 0., 1.))
    # plt.show()

    # test style transfer
    st_shape = c_img.shape[1:]
    # x_s = Input(shape=st_shape)
    # x_c = Input(shape=st_shape)
    st = StyleTransfer(c_img, s_img, c_img, 512)
    # st_model = Model([x_s, x_c], st([x_s, x_c]), name='style_transfer')

    st.compile(optimizer='rmsprop')

    os.makedirs('results', exist_ok=True)

    n_iter = 100000
    n_log_step = 2000
    n_eval_step = 100

    for i in range(n_iter):
        st.train_on_batch()
        if i % n_eval_step == 0:
            losses = st.test_on_batch()
            print(losses)
        if i % n_log_step == 0:
            new_img = st.predict(c_img)
            plt.imsave('results/basic_{:03}.png'.format(i // n_log_step), np.clip(new_img[0] + 0.5, 0., 1.))
            print('image saved!')
    new_img = st.predict(c_img)

    plt.imshow(np.clip(new_img[0] + 0.5, 0., 1.))
    plt.show()
    # loss, grad = st.predict(img)
    # loss, grad = st(K.variable(img))
    # print(loss)
    # print(grad)
