import cv2 as cv
import numpy
import numpy as np


def konvolucija(slika: numpy.array, jedro: np.array):

    img_copy = slika.copy()
    i_height = slika.shape[0]
    i_width = slika.shape[1]

    k_height = jedro.shape[0]
    k_width = jedro.shape[1]

    # razsirimo sliko z stevilkami ki so na robu da nimamo tezav ko gre jedro
    # cez sliko ker pixel 0,0 nima pixela levo zgoraj itd....
    slika = np.pad(slika, ((k_height // 2, k_height // 2), (k_width // 2, k_width // 2)), 'constant')

    for i in range(i_height):
        for j in range(i_width):
            img_copy[i][j] = np.sum(slika[i:i + k_height, j:j + k_width] * jedro)

    return img_copy


def filtriraj_z_gaussovim_jedrom(slika, sigma):
    kernel_size = int(2 * sigma) * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    k = (kernel_size / 2) - 0.5


    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((i - k) ** 2 + (j - k) ** 2) / (2 * sigma ** 2)) # formula za gaussovo jedro

    return konvolucija(slika, kernel)


def filtriraj_sobel_smer(slika):
    # filter image with sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    slika = konvolucija(slika, sobel_x)
    return konvolucija(slika, sobel_y)



if __name__ == '__main__':
    pass
