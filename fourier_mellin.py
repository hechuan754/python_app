import cv2
import numpy as np
import math
from copy1 import copy2
import time
from matplotlib import pyplot as plt
import ctypes


#像素值阈值分割，突出显示
def process_image(image):
    # 获取图片的像素值
    pixels = image.flatten()

    # # 对像素值进行排序
    # pixels_sorted = sorted(pixels, reverse=True)
    # # 获取前x个最亮的像素值
    # brightest_pixels = pixels_sorted[:x]

    # 将最亮的像素值设为255，其他像素值设为0
    # new_pixels = [255 if pixel in brightest_pixels else 0 for pixel in pixels]
    new_pixels = [pixel if pixel >= 245 and pixel <= 255 else 0 for pixel in pixels]
    # 将新的像素值重新写入图片
    new_pixels = np.array(new_pixels)
    fnew_img = new_pixels.reshape(image.shape)
    return fnew_img

#fft
def extract_image(image):

    # filtered_img = cv2.fastNlMeansDenoising(image, None, 50, 7, 21)
    # image = image.astype(np.float32) - filtered_img.astype(np.float32)
    f = np.fft.fft2(image)  # 快速傅里叶变换算法得到频率分布
    fshift = np.fft.fftshift(f)  # 将图像中的低频部分移动到图像的中心，默认是在左上角
    fimg = 20 * np.log(np.abs(fshift)+1e-5)
    cv2.imwrite('fft.jpg', fimg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # cv2.imwrite("1.png", fimg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # plt.subplot(111), plt.imshow(fimg, 'gray'), plt.title('Fourier Image')
    # plt.axis('off')
    # fimg = process_image(fimg)
    # cv2.imwrite('fft-h.jpg', fimg, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return fimg
##自定义旋转缩放
def rotated(img, angle, ratio):
    rows, cols = img.shape[:2]
    rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, ratio)
    dst = cv2.warpAffine(img, rotate, (cols, rows))
    #cv2.imwrite('./syn/1.png', dst)
    return dst


##对数极坐标变换
def logpolar(img):
    h, w = img.shape[:2]
    maxRadius = math.hypot(w / 2, h / 2)   # 距离
    m = w / math.log(maxRadius)
    log_polar = cv2.logPolar(img, (w / 2, h / 2), m, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
    return log_polar


# def interp2(x, y, img, xi, yi):
#     """
#     最终版 very nice
#     x, y: 初始坐标
#     img: 待插值图像
#     xi, yi： 插值图像坐标
#
#     使用双线性插值实现：
#     -----------------------------
#     | q11(x1, y1) | q12(x1, y2) |
#                 q(x,y)
#     | q21(x2, y1) | q22(x2, y2) |
#     -----------------------------
def interp2(x, y, img, xi, yi):
    """
    按照matlab interp2写的加速2d插值
    当矩阵规模很大的时候,numba就快,矩阵规模小,则启动numba有开销
    原图是规整矩阵才能这么做
    """

    # @jit
    def _interpolation(x, y, m, n, mm, nn, zxi, zyi, alpha, beta, img, return_img):
        qsx = -int(m / 2)
        qsx = -1
        qsy = int(n / 2)
        qsy = -1
        zxi[zxi < 1] = 1
        zxi[zxi > m - 1] = m - 1
        zyi[zyi < 1] = 1
        zyi[zyi > n - 1] = n - 1
        for i in range(mm):  # 行号
            for j in range(nn):
                zsx, zsy = int(zxi[i, j] + qsx), int(zyi[i, j] + qsy)  # 左上的列坐标和行坐标
                zxx, zxy = int(zxi[i, j] + qsx), int(zyi[i, j] + qsy + 1)  # 左下的列坐标和行坐标
                ysx, ysy = int(zxi[i, j] + qsx + 1), int(zyi[i, j] + qsy)  # 右上的列坐标和行坐标
                yxx, yxy = int(zxi[i, j] + qsx + 1), int(zyi[i, j] + qsy + 1)  # 右下的列坐标和行坐标
                fu0v = img[zsy, zsx] + alpha[i, j] * (img[ysy, ysx] - img[zsy, zsx])
                fu0v1 = img[zxy, zxx] + alpha[i, j] * (img[yxy, yxx] - img[zxy, zxx])
                fu0v0 = fu0v + beta[i, j] * (fu0v1 - fu0v)
                return_img[i, j] = fu0v0
        return return_img

    m, n = img.shape  # 原始大矩阵大小
    mm, nn = xi.shape  # 小矩阵大小,mm为行,nn为列
    zxi = np.floor(xi)  # 用[u0]表示不超过S的最大整数
    zyi = np.floor(yi)
    alpha = xi - zxi  # u0-[u0]
    beta = yi - zyi
    return_img = np.zeros((mm, nn))
    return_img = _interpolation(x, y, m, n, mm, nn, zxi, zyi, alpha, beta, img, return_img)
    return return_img


##高通滤波矩阵
def hipass_filter(ht, wd):
    res_ht = 1 / (ht - 1)
    res_wd = 1 / (wd - 1)

    eta = np.arange(-0.5, 0.5, res_ht)
    eta = np.append(eta, 0.5).reshape(1, ht)
    eta = np.cos(np.pi * eta)
    neta = np.arange(-0.5, 0.5, res_wd)
    neta = np.append(neta, 0.5).reshape(1, wd)
    neta = np.cos(np.pi * neta)
    X = np.dot(eta.T, neta)

    H = (1.0 - X) * (2.0 - X)
    return H

def LogPolarFFTTemplateMatch(img, angle_accuracy, scale_accuracy):
    h, w = img.shape[:2]
    ## 边缘检测
    # im0 = cv2.Canny(im0, canny_threshold1, canny_threshold2)
    # im1 = cv2.Canny(im1, canny_threshold1, canny_threshold2)

    ## 归一化
    # im0 = im0.astype(np.float32) / 255.0
    # im1 = im1.astype(np.float32) / 255.0

    ## fft变换到频域
    # F0 = np.fft.fft2(im0)
    # F0 = np.fft.fftshift(F0)
    # f0 = np.abs(F0)
    # F1 = np.fft.fft2(im1)
    # F1 = np.fft.fftshift(F1)  # 将零频率分量移至频谱中心
    # f1 = np.abs(F1)
    # HF = hipass_filter(h, w)
    #
    # ## 高通滤波
    # IA = HF * f0
    # IB = HF * f1
    # IA = cv2.imread('./syn_Image/syn-copy-720-fft.png', 0).astype(float)
    IA = cv2.imread('./syn_Image/syn-copy-1024-fft.png', 0).astype(float)
    time1 = time.time()
    IB = extract_image(img).astype(float)
    time2 = time.time()
    print("fft变化所需时长", time2 - time1)
    # IB = img

    Ntheta = angle_accuracy
    Rtheta = [0, 2 * np.pi]
    Nrho = scale_accuracy
    Rrho = [1, min(h / 2, w / 2)]
    time1 = time.time()
    ##Caculate Polar Matrix计算极坐标矩阵
    theta = np.linspace(Rtheta[0], Rtheta[1], Ntheta + 1)
    theta = np.delete(theta, -1).reshape(1, Ntheta)
    rho = np.logspace(np.log10(Rrho[0]), np.log10(Rrho[1]), Nrho).reshape(1, Nrho)
    xx = np.dot(rho.T, np.cos(theta)) + w / 2
    yy = np.dot(rho.T, np.sin(theta)) + h / 2
    x1 = np.linspace(0, w - 1, w)
    y1 = np.linspace(0, h - 1, h)
    x, y = np.meshgrid(x1, y1)
    time2 = time.time()
    print("计算极坐标所需时间", time2 - time1)


    # ll = ctypes.cdll.LoadLibrary
    # lib = ll('C:/Users/Administrator/Desktop/creat-dll/x64/Debug/creat-dll.dll')
    time1 = time.time()
    ##2维线性插值
    L1 = interp2(x, y, IA, xx, yy)
    L1[(xx > h - 1)] = 0
    L1[(xx < 1)] = 0
    L1[(yy > w - 1)] = 0
    L1[(yy < 1)] = 0
    L2 = interp2(x, y, IB, xx, yy)
    L2[(xx > h - 1)] = 0
    L2[(xx < 1)] = 0
    L2[(yy > w - 1)] = 0
    L2[(yy < 1)] = 0
    time2 = time.time()
    print("2维线性插值所需时间", time2 - time1)

    time1 = time.time()
    ##Phase Coralation相关度匹配
    FL1 = np.fft.fft2(L1)
    FL2 = np.fft.fft2(L2)

    FCross = np.exp((1j * (np.angle(FL1) - np.angle(FL2))))
    FPhase = np.real(np.fft.ifft2(FCross))
    R = FPhase
    R[:, int(angle_accuracy / 4): int(angle_accuracy / 4 * 3)] = 0
    R[int(scale_accuracy / 5): int(scale_accuracy / 5 * 4), :] = 0
    x, y = np.where(R == np.max(R))
    x = x[0]
    y = y[0]
    angle = 180.0 * theta[0][y] / np.pi
    if x > ((scale_accuracy // 2) - 1):
        x = scale_accuracy - 1 - x
        scale = 1 / rho[0][x]
    else:
        scale = rho[0][x]
    time2 = time.time()
    print("相关度匹配所需时间", time2 - time1)
    return scale, angle

def recover(I1):
    h,w = I1.shape[:2]
    # cv2.imwrite('C:/Users/jennie/Desktop/watermark/code/syn_Image/syn-copy.png', I3)
    scale, angle = LogPolarFFTTemplateMatch(I1, 720, 400)
    # if (angle > 270): angle -= 360
    time1 = time.time()
    # rotate = cv2.getRotationMatrix2D((h * 0.5, w * 0.5), angle, scale)
    # dst = cv2.warpAffine(I1, rotate, (w, h))
    # time2 = time.time()
    # print("逆旋转缩放所需时间", time2 - time1)
    # print("angle=", angle, "scale=", scale)
    return scale, angle

if __name__ == "__main__":
    # I1 = cv2.imread('C:/Users/jennie/Desktop/watermark/code1/embeded_image3/256/embed1-1024-2-2-2.png', 0)
    I1 = cv2.imread('fft.jpg', 0)
    # I1 = cv2.imread('./match/2222-3.jpg')
    # I1 = rotated(I1, 2.6, 1.3)
    scale, angle = LogPolarFFTTemplateMatch(I1, 3600, 2000)
    print("angle=", angle, "scale=", scale)
    # dst = recover(I1)
    # cv2.imshow('recover.png', dst.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])


