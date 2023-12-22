import reedsolo as rs
## 解码

def decode(msgList):
    rsc = rs.RSCodec(int(len(msgList)/2))  # 设置容错码长 25%
    resArray = bytearray(msgList)  # List转bytearray
    res = rsc.decode(resArray)  # rs解码
    # resStr = res[0].decode() # 转换成字符串输出
    resList = list(res[0])  # 转换成List输出

    ## 将十进制list转二进制 再转十进制
    decList = []
    binList = []
    for i in range(len(resList)):
        m = bin(resList[i])
        for j in range(len(m) - 1, 1, -1):
            binList.append(m[j])
        if i < len(resList) - 1 and len(m) < 10:
            for k in range(10 - len(m)):
                binList.append('0')
    res = 0
    for i in range(len(binList)):
        if binList[i] == '1':
            res += (2 ** (len(binList) - 1 - i))
    return res

# right_code = [110, 251, 71, 246, 120, 231, 227, 73, 83, 223, 252, 48, 198, 121, 108, 154]
# decode(right_code)
