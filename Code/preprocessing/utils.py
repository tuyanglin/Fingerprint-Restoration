import math
from functools import reduce


def apply_kernel_at(get_value, kernel, i, j):
    kernel_size = len(kernel)
    result = 0
    for k in range(0, kernel_size):
        for l in range(0, kernel_size):
            pixel = get_value(i + k - kernel_size // 2, j + l - kernel_size // 2)
            result += pixel * kernel[k][l]
    return result


def apply_to_each_pixel(pixels, f):
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            pixels[i][j] = f(pixels[i][j])


def flatten(ls):
    return reduce(lambda x, y: x + y, ls, [])


def transpose(ls):
    return list(map(list, zip(*ls)))


def gauss(x, y):
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))


def kernel_from_function(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel


#
# def gauss_kernel(size):
#     return kernel_from_function(size, gauss)


def apply_kernel(pixels, kernel):
    apply_kernel_with_f(pixels, kernel, lambda old, new: new)


def apply_kernel_with_f(pixels, kernel, f):
    size = len(kernel)
    for i in range(size // 2, len(pixels) - size // 2):
        for j in range(size // 2, len(pixels[i]) - size // 2):
            pixels[i][j] = f(pixels[i][j], apply_kernel_at(lambda x, y: pixels[x][y], kernel, i, j))


def load_image(im):
    (x, y) = im.shape
    # im_load = im.load()

    result = []
    for i in range(x):
        result.append([])
        for j in range(y):
            result[i].append(im[i, j])

    return result


def load_pixels(im, pixels):
    (x, y) = im.shape
    # im_load = im.load()

    for i in range(x):
        for j in range(y):
            im[i, j] = pixels[i][j]
