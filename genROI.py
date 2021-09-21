import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
import time
import cv2 as cv
import selectivesearch.selectivesearch as ss

def imconv(image_array, suanzi):
    image = image_array.copy()
    dim1, dim2 = image.shape
    for i in range(1, dim1 - 1):
        for j in range(1, dim2 - 1):
            image[i, j] = (image_array[(i - 1):(i + 2), (j - 1):(j + 2)] * suanzi).sum()
    image = image * (255.0 / image.max())
    return image

def removeNoise(img, kernel):
    s_time = time.clock()
    img_res = np.zeros_like(img)
    dim1, dim2 = img.shape
    res_list = []
    img /= 255
    zeros_r = np.ones((12, img.shape[1]))
    zeros_c = np.ones((408, 12))
    padding = np.concatenate((zeros_r, img), axis=0)
    padding = np.concatenate((padding, zeros_r), axis=0)
    padding = np.concatenate((zeros_c, padding), axis=1)
    padding = np.concatenate((padding, zeros_c), axis=1)
    for i in range(12, dim1 - 12, 1):
        for j in range(12, dim2 - 12, 1):
            res_list.append((padding[(i - 12): (i + 13),
                             (j - 12): (j + 13)] * kernel).sum())
            if res_list[-1] <= 550:
                img_res[i - 12][j - 12] = 1


    img_res *= 255
    plt.figure()
    plt.imshow(img_res, cmap=cm.gray)
    plt.savefig('bin gray3.png')
    plt.show()

    return img_res

def detect(img_path):

    suanzi1 = np.array([[0, 1, 0],
                        [1,-4, 1],
                        [0, 1, 0]])

    suanzi2 = np.array([[1, 1, 1],
                        [1,-8, 1],
                        [1, 1, 1]])

    image = Image.open(img_path).convert("L")
    image_array = np.array(image)
    image_array.resize(384, 512)
    image_suanzi1 = signal.convolve2d(image_array,suanzi1,mode="same")
    image_suanzi2 = signal.convolve2d(image_array,suanzi2,mode="same")

    image_suanzi1 = (image_suanzi1/float(image_suanzi1.max()))*255
    image_suanzi2 = (image_suanzi2/float(image_suanzi2.max()))*255


    image_suanzi1[image_suanzi1>image_suanzi1.mean()] = 255
    image_suanzi2[image_suanzi2>image_suanzi2.mean()] = 255

    return image_suanzi1, image_suanzi2

def calculate(img):
    return img.sum()

def selective_search(im, mode='fast'):
    cv.setUseOptimized(True)
    cv.setNumThreads(4)
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)

    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    elif mode == 'quality':
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    return rects

def getROI(img, img_path):
    dim1, dim2 = img.shape

    img /= 255
    origin_img = cv.imread(img_path)

    regions = selective_search(origin_img, mode='fast')
    sum_pic = dim1 * dim2 - img.sum()
    max_Sq, max_region = 0, [0] * 4
    print(len(regions))
    for r in regions:
        x, y, h, w = r
        temp_Sq = img[x: x+w, y: y+h].sum()
        temp_Sq = h * w - temp_Sq
        if temp_Sq >= max_Sq and temp_Sq != sum_pic:
            max_region = x, y, h, w
            max_Sq = temp_Sq
    print(max_region[2] * max_region[3])
    return max_region

def getROI_1(img, img_path):
    dim1, dim2 = img.shape

    img /= 255
    origin_img = cv.imread(img_path)

    _, regions = ss.selective_search(origin_img, scale=1000,
                                     sigma=0.8, min_size=10000)
    sum_pic = dim1 * dim2 - img.sum()
    max_Sq, max_region = 0, [0] * 4
    print(len(regions))
    for r in regions:
        x, y, h, w = r['rect']
        temp_Sq = img[x: x+h, y: y+w].sum()
        temp_Sq = h * w - temp_Sq
        if temp_Sq >= max_Sq and temp_Sq <= 0.98 * sum_pic:
            max_region = x, y, h, w
            max_Sq = temp_Sq
    print(max_region[2] * max_region[3])
    return max_region

def draw(img_path, region):
    Roi = region
    x, y, h, w = Roi
    pic = cv.imread(img_path)

    cv.rectangle(pic, (x, y), (x + h, y + w), (125, 0, 0), 3)
    plt.figure()
    plt.imshow(pic)
    plt.savefig('Region with pic2.png')
    plt.show()


def main():
    image_suanzi1, image_suanzi2 = detect(img_path)
    img_res = removeNoise(image_suanzi2, noise_kernel)
    x, y, h, w = getROI_1(img_res, img_path)
    draw(img_path, (x, y, h, w))

if __name__ == '__main__':
    noise_kernel = np.ones((25, 25))
    img_path = ""
    main()