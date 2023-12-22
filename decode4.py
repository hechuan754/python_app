import numpy as np
from scipy.linalg import hadamard
import cv2
from rscode import rsDecode

def decode(img):
    # np.set_printoptions(threshold = 1e6)
    imgLength = 64
    expandTimes = 64
    codeLen = 256  # 码进制长度
    lenRsCode = 16  ##记录rs编码数据的长度
    H4 = hadamard(codeLen)
    # print(H4)
    matrix1 = np.ones((codeLen, codeLen))
    matrix2 = np.ones((imgLength, expandTimes))
    matrixGet = np.ones((lenRsCode, codeLen))
    matrixGet2 = np.ones((imgLength, expandTimes))
    imgGet2 = np.ones((imgLength*2, expandTimes*2))

    # print(matrix1)

    col = 0
    # codeList = [245, 162, 144, 12, 79, 236, 156, 79, 62, 238, 185, 151, 11, 147, 25, 160]
    # codeList = [170, 94, 244, 50, 255, 54, 100, 147, 112, 139, 215, 105, 111, 247, 231, 96]
    # codeList = [123, 96, 71, 118, 6, 212, 218, 15, 69, 244, 50, 174, 142, 19, 246, 107]
    codeList = [123, 96, 71, 246, 120, 231, 222, 15, 190, 223, 252, 48, 198, 121, 108, 154]
    codeMatrix = np.ones((imgLength * 2, expandTimes * 2))
    # Press the green button in the gutter to run the script.

    for i in range(0, codeLen):
        matrix1[col] = H4[i]
        col += 1
    print(matrix1[1])
    print(H4[1])
    # #图像缩小
    for i in range(128):
        for j in range(128):
            tempCnt = 0
            for k in range(2):
                for m in range(2):
                    tempCnt += img[i * 2 + k][j * 2 + m]
            tempCnt /= 4
            imgGet2[i][j] = int(tempCnt)

    # # 生成随机映射list
    listRandom = [i for i in range(imgLength * 2 * expandTimes * 2)]
    # # 不采用随机种子，采用自定义种子
    rng = np.random.default_rng(31415926)
    rng.shuffle(listRandom)

    # # 将codeMatrix反向恢复映射
    codeMatrixRandom = np.ones((imgLength * 2, expandTimes * 2))
    for i in range(imgLength * 2):
        for j in range(expandTimes * 2):
            xIndex = int(listRandom[i * expandTimes * 2 + j] / (expandTimes * 2))
            yIndex = listRandom[i * expandTimes * 2 + j] % (imgLength * 2)
            codeMatrixRandom[xIndex][yIndex] = imgGet2[i][j]

    # 2d解码
    for i in range(imgLength):
        for j in range(expandTimes):
            tempCnt = 0
            for k in range(2):
                for m in range(2):
                    tempCnt += codeMatrixRandom[i * 2 + k][j * 2 + m]
            tempCnt /= 4
            if tempCnt > 100:
                matrixGet2[i][j] = 1
            else:
                matrixGet2[i][j] = -1
    #马赛克解码
    # for i in range(imgLength):
    #     for j in range(expandTimes):
    #         tempCnt = 0
    #         for k in range(2):
    #             for m in range(2):
    #                 tempCnt += codeMatrixRandom[i * 2 + k][j * 2 + m]
    #         tempCnt /= 4
    #         if tempCnt > 105:
    #             matrixGet2[i][j] = 1
    #         else:
    #             matrixGet2[i][j] = -1
    # # # 块状存储恢复成条状
    # xx = 16
    # yy = 16
    # for i in range(int(imgLength)):
    #     for j in range(int(expandTimes)):
    #         index1 = int(i/xx) * xx + int(((i*expandTimes+j)%(xx*yy))/yy)
    #         index2 = (i % xx) * yy + ((i*expandTimes+j)%(xx*yy))%yy
    #         matrixGet[i][j] = matrixGet2[index1][index2]
    #
    for i in range(int(imgLength)):
        for j in range(int(expandTimes)):
            index1 = int((i * expandTimes + j) / codeLen)
            index2 = (i * expandTimes + j) % codeLen
            matrixGet[index1][index2] = matrixGet2[i][j]

    codeListDecode = []
    ##计算内积，取最大的内积
    for i in range(lenRsCode):
        maxDot = -1
        numDecode = 0
        for j in range(codeLen):
            dot = np.dot(matrixGet[i], matrix1[j])
            # print(dot)
            if dot > maxDot:
                maxDot = dot
                numDecode = j
        codeListDecode.append(numDecode)
    #     print(codeListDecode)
    #     print("====")
    print(codeListDecode)

    rate = 0
    for i in range(len(codeListDecode)):
        if codeListDecode[i] == codeList[i]:
            rate += 1
    rate = rate / len(codeListDecode)
    print("decode rate:", rate)

    return codeListDecode
    # ##将解码数据转换回rs编码数据(多遍嵌入采用交叉验证)
    # rsGet = [0 for i in range(lenRsCode)]
    # for i in range(len(codeListDecode)):
    #     rsGet[i % lenRsCode] += codeListDecode[i]
    # for j in range(lenRsCode):
    #     rsGet[j] = int(rsGet[j]/(int(len(codeListDecode)/lenRsCode)))
    # print(rsGet)
if __name__ == '__main__':
    img = cv2.imread('./code_syn/3-256/c&s3_256_g2.png', 0)
    # img = cv2.imread('./markRandom/markRandom2.png', 0)
    # img = wieners(img, 7)
    # cv2.imwrite('./decode-wienie/1-512.png', img)
    # img = img[0:128, 0:128]
    print(decode(img))
##缩放
    # res = cv2.resize(img, (int(imgLength*2*0.5), int(expandTimes*2*0.5)), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite("/Users/hongluning/PycharmProjects/digitMark/codeDocode/resize1.png", res)
    #
    # img = cv2.imread("/Users/hongluning/PycharmProjects/digitMark/codeDocode/resize1.png", 0)
    # res = cv2.resize(img, (imgLength*2, expandTimes*2), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite("/Users/hongluning/PycharmProjects/digitMark/codeDocode/resize2.png", res)
#
# ###旋转
#     img = cv2.imread("/Users/hongluning/PycharmProjects/digitMark/codeDocode/mark.png", 0)
#     (h, w) = img.shape[:2]  # 10
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, 30, 1.0)  # 12
#     rotated = cv2.warpAffine(img, M, (w, h))  # 13
#     # cv2.imshow("Rotated by 45 Degrees", rotated)
#     cv2.imwrite("/Users/hongluning/PycharmProjects/digitMark/codeDocode/rotate0.png", rotated)
#
#     M = cv2.getRotationMatrix2D(center, -30, 1.0)  # 12
#     rotated2 = cv2.warpAffine(rotated, M, (w, h))  # 13
#     # cv2.imshow("Rotated by -45 Degrees", rotated2)
#     cv2.imwrite("/Users/hongluning/PycharmProjects/digitMark/codeDocode/rotate1.png", rotated2)

    ##反向解出数字
    #img = cv2.imread("markRandom_gauss.png", 0)
