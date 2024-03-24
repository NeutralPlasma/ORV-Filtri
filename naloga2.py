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
    '''Filtrira sliko z Gaussovim jedrom..'''
    pass


def filtriraj_sobel_smer(slika):
    '''Filtrira sliko z Sobelovim jedrom in označi gradiente v orignalni sliki glede na ustrezen pogoj.'''
    pass


if __name__ == '__main__':
    pass
