import socket
from PIL import Image
import os
import imghdr
import base64
from io import BytesIO

tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

address = ("192.168.194.67", 1213)  # 绑定IP地址和端口
tcp_server.bind(address)

tcp_server.listen(5)
print("服务器已启动，等待客户端连接")

counter = 1

while True:
    tcp_client, addr = tcp_server.accept()
    print('客户端连接地址', addr)

    # 接收图片大小信息
    expected_size_data = tcp_client.recv(1024)
    expected_size_str = expected_size_data.decode('utf-8').strip()

    if not expected_size_str:
        continue

    expected_size = int(expected_size_str)

    # 接收Base64编码的图像数据
    data = b''

    while len(data) < expected_size:
        chunk = tcp_client.recv(8192)
        if not chunk:
            break
        data += chunk

    # 解码Base64数据
    image_data = base64.b64decode(data)

    temp_file = BytesIO(image_data)

    # 将 BytesIO 对象中的数据直接解码为图像
    image = Image.open(temp_file)

    temp_dir = "./data/img"
    os.makedirs(temp_dir, exist_ok=True)
    image_filename = f"image_{counter}.jpg"
    image_path = os.path.join(temp_dir, image_filename)

    image_format = imghdr.what(None, h=image_data)

    if image_format:
        image.save(image_path, format=image_format)
        print(f"图像保存成功: {image_filename}")
        counter += 1
    else:
        print("未能识别图像格式")

    tcp_client.close()
