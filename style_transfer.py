from src.core import *

import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content-image', type=str)
    parser.add_argument('--style-image', type=str)
    parser.add_argument('--content-weight', type=float, default=1)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    args = parser.parse_args()
    loss, stylized_image = run(args.style_image, args.content_image, args.content_weight * 16, max_scale=4)

    plt.imsave(os.path.join('results', 'output.png'), stylized_image[0])


if __name__ == '__main__':
    main()
