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
    slika = np.pad(slika, ((k_height // 2, k_height // 2), (k_width // 2, k_width // 2)), 'edge')
    pad_x = k_height // 2
    pad_y = k_width // 2

    for i in range(i_height):
        for j in range(i_width):
            i_moved = i + pad_x
            j_moved = j + pad_y

            x_start = i
            x_end = i_moved + pad_x + 1

            y_start = j
            y_end = j_moved + pad_y + 1

            img_copy[i][j] = np.sum(slika[x_start:x_end, y_start:y_end] * jedro)

    return img_copy


def filtriraj_z_gaussovim_jedrom(slika, sigma):
    kernel_size = int(2 * sigma) * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    k = (kernel_size / 2) - 0.5

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = 1 / (2 * np.pi * sigma ** 2) * np.exp(
                -((i - k) ** 2 + (j - k) ** 2) / (2 * sigma ** 2))  # formula za gaussovo jedro

    return konvolucija(slika, kernel)


def filtriraj_sobel_smer(slika):
    # filter image with sobel filter
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * (1/9)
    sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * (1/9)

    sobel_x = konvolucija(slika, sobel_x_kernel)
    sobel_y = konvolucija(slika, sobel_y_kernel)

    output = np.zeros_like(slika)

    # join sobel x and sobel y
    output = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    output *= 255.0 / output.max()
    output = np.uint8(output)

    #for i in range(sobel_x.shape[0]):
    #    for j in range(sobel_x.shape[1]):
    #        output[i][j] = (sobel_x[i][j] ** 2) + (sobel_y[i][j] ** 2)


    return output


if __name__ == '__main__':
    slika = cv.imread(".utils/lenna.png")
    slika = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    slika = filtriraj_z_gaussovim_jedrom(slika, 1)

    sobel_orig = filtriraj_sobel_smer(slika)

    cv.imshow("GrayScale", slika)
    cv.imshow("Sobel orig", sobel_orig)

    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])



    cv.waitKey(0)
    cv.destroyAllWindows()
