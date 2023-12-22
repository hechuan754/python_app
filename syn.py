import cv2
import numpy as np
from decode4 import decode
import gray
from scipy.signal import wiener
from gray import gray
from rscode import rsDecode
from fourier_mellin import recover
import time


def median(img):
    # 读入图像并转换为灰度图像
    # 将图像像素值转换为一维数组
    pixel_values = np.array(img).flatten()
    # 对数组进行排序
    sorted_pixel_values = np.sort(pixel_values)
    # 找到中位数
    med = np.median(sorted_pixel_values)
    print('图像的像素值中位数为:', med)
    return med

def wiener_RGB(img):
    # gray(img)
    b, g, r = cv2.split(img)
    b_s = 2
    g_s = 2
    r_s = 2
    b_wiener = wiener(b, [6, 6])
    g_wiener = wiener(g, [6, 6])
    r_wiener = wiener(r, [6, 6])
    # b_copy = (b.astype(np.float32) - b_wiener.astype(np.float32)) / b_s
    # g_copy = (g.astype(np.float32) - g_wiener.astype(np.float32)) / g_s
    # r_copy = (r.astype(np.float32) - r_wiener.astype(np.float32)) / r_s
    b_copy = (b.astype(np.float32) - b_wiener) * 2
    g_copy = (g.astype(np.float32) - g_wiener) * 2
    r_copy = (r.astype(np.float32) - r_wiener) * 2
    # cv2.imshow('b', b_copy.astype(np.uint8))
    # cv2.imshow('g', g_copy.astype(np.uint8))
    # cv2.imshow('r', r_copy.astype(np.uint8))
    # img = (r_wiener - g_wiener + b_wiener) + 120
    # img = b_copy
    # gray(b_copy)
    img = (b_copy + r_copy + g_copy) / 3
    img = 120 - median(img) + img
    # img = J
    # gray(img)
    # b_copy += 100
    img[img < 0] = 0
    img[img > 255] = 255
    # gray(img)
    cv2.imshow('wiener', img.astype(np.uint8))
    cv2.waitKey()
    # J = cv2.cvtColor(J, cv2.COLOR_YUV2RGB)
    # J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)

def syn(origin):
    # template = cv2.imread("./syn_Image/syn-105.png", 0)
    template = cv2.imread("./syn_Image/syn_copy_256.png", 0)
    # template = cv2.imread("./markRandom/markRandom2.png")
    # template = cv2.imread("C:/Users/jennie/Desktop/watermark/code1/syn_Image/syn_expand_256.png", 0)
    # origin是嵌入水印后的彩色图片，经过维纳滤波
    # target是阈值分割后的结果
    time1 = time.time()
    wiener_img = wiener_RGB(origin)
    wiener_img = origin.copy()
    # cv2.imwrite('wiener.jpg', wiener_img)
    time2 = time.time()
    print("滤波时长：", time2 - time1, 's')
    print("""""")

    time3 = time.time()

    scale, angle = recover(wiener_img)

    time1 = time.time()
    rotate = cv2.getRotationMatrix2D((h * 0.5, w * 0.5), angle, scale)
    wiener_img = cv2.warpAffine(wiener_img, rotate, (w, h))

    time2 = time.time()
    print("逆旋转缩放所需时间", time2 - time1)

    time4 = time.time()
    print("矫正总时长：", time4 - time3, 's')
    print("""""")
    # origin = origin.astype(np.uint8) + 120
    # gray(origin)
    # gray(target)
    wiener_img = origin.copy()
    target = wiener_img.copy()
    gray(target)
    # med = median(target)
    # target[target <= 122] = 0
    # target[target > 122] = 255
    # wiener_img = target.copy()
    # med = median(target)
    # cv2.imshow('1', target)
    # cv2.waitKey()
    # gray(target)
    # target = target.astype(np.uint8)
    # cv2.imshow('1',target)
    # opencv模板匹配----多目标匹配

    theight, twidth = target.shape[:2]
    # theight = 128
    # twidth = 128
    # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
    time5 = time.time()
    result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
    # 归一化处理
    # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    # 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # 绘制矩形边框，将匹配区域标注出来
    # min_loc：矩形定点
    # (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
    # (0,0,225)：矩形的边框颜色；2：矩形边框宽度
    cv2.rectangle(wiener_img, min_loc, (min_loc[0] + 256, min_loc[1] + 256), (255, 0, 0), 1)
    print(min_loc[0], min_loc[1])
    #0是横的宽度，i是竖的高度
    flag = False
    if flag is False:
        crop = wiener_img[min_loc[1]:min_loc[1] + 256, min_loc[0]:min_loc[0] + 256]
        codeListDecode = decode(crop)
        try:
            if rsDecode.decode(codeListDecode) == right_res:
                # resStr = rsDecode.decode(codeListDecode)
                # if resStr == right_res:
                flag = True
                print("解码结果：", right_res)
                cv2.rectangle(wiener_img, min_loc, (min_loc[0] + 256, min_loc[1] + 256), (0, 0, 225), 1)
        except:
            print('')
    if flag is False:
        if min_loc[1] - 128 >= 0:
            crop = wiener_img[min_loc[1] - 128:min_loc[1] + 128, min_loc[0]:min_loc[0] + 256]
            # cv2.rectangle(origin, min_loc, (min_loc[0] + 256, min_loc[1] + 256), (255, 0, 0), 1)
            # # cv2.imwrite('1.png', crop)
            # # print('最佳匹配结果：')
            codeListDecode = decode(crop)
            try:
                if rsDecode.decode(codeListDecode) == right_res:
                    # resStr = rsDecode.decode(codeListDecode)
                    # if resStr == right_res:
                    flag = True
                    print("解码结果：", right_res)
                    cv2.rectangle(wiener_img, (min_loc[0], min_loc[1] - 128), (min_loc[0] + 256, min_loc[1] + 128), (0, 0, 225), 1)
            except:
                print('')
    if flag is False:
        if min_loc[1] - 128 >= 0 and min_loc[0] - 128 >= 0:
            crop = wiener_img[min_loc[0] - 128:min_loc[0] + 128, min_loc[1] - 128:min_loc[1] + 128]
            # cv2.rectangle(origin, min_loc, (min_loc[0] + 256, min_loc[1] + 256), (255, 0, 0), 1)
            # # cv2.imwrite('1.png', crop)
            # # print('最佳匹配结果：')
            codeListDecode = decode(crop)
            try:
                if rsDecode.decode(codeListDecode) == right_res:
                    # resStr = rsDecode.decode(codeListDecode)
                    # if resStr == right_res:
                    flag = True
                    print("解码结果：", right_res)
                    cv2.rectangle(wiener_img, (min_loc[0] - 128, min_loc[1] - 128), (min_loc[0] + 128, min_loc[1] + 128), (0, 0, 225), 1)
            except:
                print('')
    if flag is False:
        if min_loc[0] - 128 >= 0:
            crop = wiener_img[min_loc[1]:min_loc[1] + 256, min_loc[0] - 128:min_loc[0] + 128]
            # cv2.rectangle(origin, min_loc, (min_loc[0] + 256, min_loc[1] + 256), (255, 0, 0), 1)
            # # cv2.imwrite('1.png', crop)
            # # print('最佳匹配结果：')
            codeListDecode = decode(crop)
            try:
                if rsDecode.decode(codeListDecode) == right_res:
                    # resStr = rsDecode.decode(codeListDecode)
                    # if resStr == right_res:
                    flag = True
                    print("解码结果：", right_res)
                    cv2.rectangle(wiener_img, (min_loc[0] - 128, min_loc[1]), (min_loc[0] + 128, min_loc[1] + 256), (0, 0, 225), 1)
            except:
                print('')
    if flag is False:
        if min_loc[1] + 384 <= theight and min_loc[0] - 128 >= 0:
            crop = wiener_img[min_loc[1] + 128:min_loc[1] + 384, min_loc[0] - 128:min_loc[0] + 128]
            codeListDecode = decode(crop)
            try:
                if rsDecode.decode(codeListDecode) == right_res:
                    flag = True
                    print("解码结果：", right_res)
                    cv2.rectangle(wiener_img, (min_loc[0] - 128, min_loc[1] + 128), (min_loc[0] + 128, min_loc[1] + 384), (0, 0, 225), 1)
            except:
                print('')
    if flag is False:
        if min_loc[1] + 384 <= theight:
            crop = wiener_img[min_loc[1] + 128:min_loc[1] + 384, min_loc[0]:min_loc[0] + 256]
            codeListDecode = decode(crop)
            try:
                if rsDecode.decode(codeListDecode) == right_res:
                    flag = True
                    print("解码结果：", right_res)
                    cv2.rectangle(wiener_img, (min_loc[0], min_loc[1] + 128), (min_loc[0] + 256, min_loc[1] + 384), (0, 0, 225), 1)
            except:
                print('')
    if flag is False:
        if min_loc[1] + 384 <= theight and min_loc[0] + 384 <= twidth:
            crop = wiener_img[min_loc[1] + 128:min_loc[1] + 384, min_loc[0] + 128:min_loc[0] + 384]
            codeListDecode = decode(crop)
            try:
                if rsDecode.decode(codeListDecode) == right_res:
                    flag = True
                    print("解码结果：", right_res)
                    cv2.rectangle(wiener_img, (min_loc[0] + 128, min_loc[1] + 128), (min_loc[0] + 384, min_loc[1] + 384), (0, 0, 225), 1)
            except:
                print('')
    if flag is False:
        if min_loc[0] + 384 <= twidth:
            crop = wiener_img[min_loc[1]:min_loc[1] + 256, min_loc[0] + 128:min_loc[0] + 384]
            codeListDecode = decode(crop)
            try:
                if rsDecode.decode(codeListDecode) == right_res:
                    flag = True
                    print("解码结果：", right_res)
                    cv2.rectangle(wiener_img, (min_loc[0] + 128, min_loc[1]), (min_loc[0] + 384, min_loc[1] + 256), (0, 0, 225), 1)
            except:
                print('')
    if flag is False:
        if min_loc[1] - 128 >= 0 and min_loc[0] + 384 <= twidth:
            crop = wiener_img[min_loc[1] - 128:min_loc[1] + 128, min_loc[0] + 128:min_loc[0] + 384]
            codeListDecode = decode(crop)
            try:
                if rsDecode.decode(codeListDecode) == right_res:
                    flag = True
                    print("解码结果：", right_res)
                    cv2.rectangle(wiener_img, (min_loc[0] + 128, min_loc[1] - 128), (min_loc[0] + 384, min_loc[1] + 128), (0, 0, 225), 1)
            except:
                print('')
    # 匹配值转换为字符串
    # 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
    # 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
    strmin_val = str(min_val)
    # 初始化位置参数
    # temp_loc = [0, 0]
    # other_loc = min_loc
    numOfloc = 0
    # 第一次筛选----规定匹配阈值，将满足阈值的从result中提取出来
    # 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法设置匹配阈值为0.01
    # print('其他匹配结果：')
    # threshold = 0.98
    # loc = np.where(result < threshold)
    # # # 遍历提取出来的位置
    # for other_loc in zip(*loc[::-1]):
    # # #     # 第二次筛选----将位置偏移小于5个像素的结果舍去
    #     if (temp_loc[0] + 5 < other_loc[0]) or (temp_loc[1] + 5 < other_loc[1]):
    #         numOfloc = numOfloc + 1
    #         temp_loc = other_loc
    #         crop = origin[other_loc[1]:other_loc[1] + 2 * theight, other_loc[0]:other_loc[0] + 2 * twidth]
    #         cv2.rectangle(origin, other_loc, (other_loc[0] + 2*twidth, other_loc[1] + 2*theight), (255, 0, 0), 1)
    #         # decode(crop)
    #         codeListDecode = decode(crop)
    #         try:
    #             if rsDecode.decode(codeListDecode):
    #             # resStr = rsDecode.decode(codeListDecode)
    #             # if resStr == right_res:
    #                 print("解码结果：", rsDecode.decode(codeListDecode))
    #                 cv2.rectangle(origin, other_loc, (other_loc[0] + twidth, other_loc[1] + theight), (0, 0, 225), 1)
    #         except:
    #             continue

    # cv2.imwrite('./match/t1-2.jpg', origin)
    # gray(origin)
    str_numOfloc = str(numOfloc)
    # 显示结果,并将匹配值显示在标题栏上
    strText = "MatchResult----MatchingValue=" + strmin_val + "----NumberOfPosition=" + str_numOfloc + '.png'
    cv2.imshow(strText, wiener_img)
    time6 = time.time()
    print("匹配+解码时长：", time6 - time5, 's')
    print("""""")
    # cv2.imwrite('./match/1.jpg', wiener_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # time7 = time.time()
    # 读取目标图片
    right_res = 999920230822999999
    # [145, 54, 112, 228, 68, 122, 26, 255, 196, 168, 149, 76, 147, 105, 127, 216]
    # origin = cv2.imread('./photo/3-2-2.jpg')
    # filename = './photo/1'
    # filename = './embeded_image3/10.11print/222'
    # origin = cv2.imread(filename + '.jpg')
    # origin = cv2.imread('./code_syn/3-256/COPY-LADY1.png')
    # origin = cv2.imread('./embeded_image3/256/syn/LADY-2-2-2.jpg')
    # origin = cv2.imread('./embeded_image3/10.31print/LADY-1.5.jpg')
    #### 外接摄像头
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # frame_interval = 70  # 每十秒读取一帧图像
    # last_frame_time = time.time()
    # i = 1
    # while True:
    #     current_time = time.time()
    #     first_time = current_time - last_frame_time
    #     # print(current_time - last_frame_time)
    #     ret, frame = cap.read()
    #
    #     if first_time >= 3.00:
    #         print('第 1 张')
    #         # last_frame_time = current_time
    #         # cv2.imshow('Camera1', frame)
    #         frame = frame[0: 720, 0: 720, :]
    #         cv2.imwrite('Camera.jpg', frame)
    #         # origin = frame[0: 480, 0: 480, :]
    #         img = cv2.imread('Camera.jpg')
    #         syn(img)
    #         break
    #
    # while True:
    #     current_time = time.time()
    #     elapsed_time = current_time - last_frame_time
    #     ret, frame = cap.read()
    #
    #     if elapsed_time >= frame_interval:
    #         last_frame_time = current_time
    #         # print(frame.shape[:2])
    #         i += 1
    #         print('第', i, '张')
    #         frame = frame[0: 720, 0: 720, :]
    #         cv2.imwrite('Camera.jpg', frame)
    #         # print(frame.shape[:2])
    #         syn(frame)
    #
    #     # 按下'q'键退出循环
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    ####
    # current_time = time.time()
    # elapsed_time = current_time - last_frame_time
    # if elapsed_time >= frame_interval:
    # origin = camera_capture()
    # last_frame_time = current_time
    # imgcrop = origin[0: 480, 0: 480, :]
    # origin = cv2.imread('./embeded_image3/10.31print/LADY-2.jpg')
    origin = cv2.imread("C:/Users/Administrator/Desktop/fm/2.jpg", 0)
    # origin = cv2.imread('./photo/10.31/2.5-9.jpg')
    h, w = origin.shape[:2]
    #
    imgcrop = origin[h//2 - 512: h//2 + 512, w//2 - 512: w//2 + 512]
    # imgcrop = origin[h // 2 - 360: h // 2 + 360, w // 2 - 360: w // 2 + 360]
    # imgcrop = origin[0: 720, 0: 720]
    # cv2.imwrite('1.jpg', imgcrop)
    # # origin = origin[1500: 2000, 500: 1500]
    # # cv2.waitKey()
    # # origin = cv2.resize(origin, (1024, 1024))
    # # origin = rotated(origin, 20.4, 1.2)
    syn(imgcrop)
    # time8 = time.time()
    # print("总时长：", time8 - time7, 's')
    # print("""""")
    # gray(imgcrop)
