from keras.layers import *
from keras.applications import VGG16


def vgg16_pt(input_shape):
    vgg = VGG16(weights='imagenet')
    extracted_layers = [1, 2, 4, 5, 7, 8, 9, 13, 17]

    x = Input(shape=input_shape)

    outputs = []
    for l in extracted_layers:
        out = x
        for i in range(l):
            out = vgg.layers[i](out)
        outputs.append(out)
    
    # codes corresponding with `forward_cat`


def dec_lap_pyr(X, levels):
    pass

def syn_lap_pyr(pyr):
    pass


def build_style_transfer():

    cnn = vgg16_pt()

    phi = lambda x: cnn(x)


