import imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt


def laplacian_pyramid(imgs, steps: int=1):
    if len(imgs.shape) != 4:
        raise AssertionError('the image for calculate Laplacian pyramid must be 4, received {}'.format(imgs.ndim))
    
    hf, wf = max(1, 2*(steps - 1)), max(1, 2*(steps - 1))
    h, w = imgs.shape[1] // hf, imgs.shape[2] // wf
    
    results = []
    for img in imgs:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        tmp = cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
        tmp = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_LINEAR)

        n = cv2.subtract(img, tmp)
        results.append(n)
        
    return np.array(results)


def run():
    pass


if __name__ == '__main__':
    img = imageio.imread('images/butterfly.jpg')
    
    lap = laplacian_pyramid(img[np.newaxis, :])

    plt.imshow(lap[0])
    plt.show()
