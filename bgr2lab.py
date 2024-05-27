import numpy as np
import cv2


def bgr2lab(img, white_point=None):
    """
    将CV2给出的bgr格式转化为lab格式

    :param img: 输入图像，格式为gbr
    :param white_point: 输入当前光源白点的XYZ值
    :return:
    """
    # bgr->rgb并归一化
    if white_point is None:
        white_point = [95.047, 100.0, 108.883]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为rgb
    img_rgb_float = np.float64(img_rgb) * (1. / 255)  # 归一化

    # rgb->XYZ
    # gamma函数
    img_rgb_mask1 = np.where(img_rgb_float > 0.0405)
    img_rgb_mask2 = np.where(img_rgb_float <= 0.0405)

    mask1 = img_rgb_float.copy() + 0.055
    mask2 = img_rgb_float.copy()

    mask1[img_rgb_mask2] = 0
    mask2[img_rgb_mask1] = 0

    img_RGB_1 = np.power(mask1 * (1. / 1.055), 2.4)
    img_RGB_2 = mask2 * (1. / 12.92)

    img_RGB = (img_RGB_1 + img_RGB_2) * 100

    # 矩阵相乘，得到XYZ
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], dtype=np.float64)
    img_XYZ = np.einsum('ijk,kl->ijl', img_RGB, M.T)
    img_XYZ = np.around(img_XYZ, decimals=4)

    # XYZ->Lab
    img_XYZ[:, :, 0] = np.array(img_XYZ[:, :, 0] * (1. / white_point[0]), dtype=np.float64)
    img_XYZ[:, :, 1] = np.array(img_XYZ[:, :, 1] * (1. / white_point[1]), dtype=np.float64)
    img_XYZ[:, :, 2] = np.array(img_XYZ[:, :, 2] * (1. / white_point[2]), dtype=np.float64)

    img_XYZ_mask1 = np.where(img_XYZ > pow(6 / 29, 3))
    img_XYZ_mask2 = np.where(img_XYZ <= pow(6 / 29, 3))

    XYZ_mask1 = img_XYZ.copy()
    XYZ_mask2 = img_XYZ.copy()

    img_Lab_f1 = np.power(XYZ_mask1, 1 / 3)
    img_Lab_f2 = 1 / 3 * pow(29 / 6, 2) * XYZ_mask2 + 4 / 29

    img_Lab_f1[img_XYZ_mask2] = 0
    img_Lab_f2[img_XYZ_mask1] = 0

    img_Lab_f = img_Lab_f1 + img_Lab_f2

    img_Lab = np.zeros(img_XYZ.shape)

    img_Lab[:, :, 0] = 116 * img_Lab_f[:, :, 1] - 16
    img_Lab[:, :, 1] = 500 * (img_Lab_f[:, :, 0] - img_Lab_f[:, :, 1])
    img_Lab[:, :, 2] = 200 * (img_Lab_f[:, :, 1] - img_Lab_f[:, :, 2])

    return img_Lab