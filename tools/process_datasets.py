# -*- coding: utf-8 -*-
"""
@Time ： 2023/06/21 11:52
@Auth ： RS迷途小书童
@File ：Read Image.py
@IDE ：PyCharm
@Purpose：读取图片信息
@Web：博客地址:https://blog.csdn.net/m0_56729804
"""
import csv
import os
import pandas as pd
import exifread
from osgeo import gdal


def write_csv(base_path, scene=12):
    """
    读取文件夹下所有图片的信息，并将其写入到csv文件中
    """
    scene_path = os.path.join(base_path, str(scene))
    drone_path = os.path.join(scene_path, "drone")
    # CSV 文件路径
    csv_file_path = os.path.join(scene_path, "{}.csv".format(scene))
    num=1
    for filename in os.listdir(drone_path):

        img_path = os.path.join(drone_path, filename)
        # 获取图片偏航角
        print("----------------------------------大疆exifread信息---------------------------------")
        # 定义字节模式 b 和 a，用于查找大疆EXIF数据的起始和结束标记
        b = b"\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"
        a = b"\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"
        # 打开图片文件，以二进制模式读取
        img = open(img_path, 'rb')
        # 初始化一个字节数组用于存储EXIF数据
        data = bytearray()
        # 初始化一个标志，用于判断是否已经找到EXIF数据的起始标记
        flag = False
        # 逐行读取图片文件内容
        for line in img.readlines():
            # 如果当前行包含EXIF数据的起始标记，则设置标志为True
            if a in line:
                flag = True
                # 如果标志为True，则将当前行添加到EXIF数据中
            if flag:
                data += line
                # 如果当前行包含EXIF数据的结束标记，则跳出循环
            if b in line:
                break
                # 如果提取到的EXIF数据不为空
        dj_data_dict = {}
        # 遍历过滤后的行，并提取键值对存入字典中
        if len(data) > 0:
            # 将字节数据解码为ASCII字符串
            data = str(data.decode('ascii'))
            # 过滤出包含drone-dji的行，并分割每行为键值对
            lines = list(filter(lambda x: 'drone-dji:' in x, data.split("\n")))
            # 初始化一个空字典用于存储提取到的数据
            for d in lines:
                d = d.strip()[10:]  # 去除每行的前后空格和'\n'字符，并从第10个字符开始处理（因为drone-dji:占据了前9个字符）
                k, v = d.split("=")  # 将当前行分割为键和值两部分
                print(f"{k} : {v}")  # 打印键和值
                dj_data_dict[k] =  v[1:-1]  # 将键值对存入字典中

        f = open(img_path, 'rb')
        tags = exifread.process_file(f)

        # ## 焦距
        # focal = tags['EXIF FocalLength'].values[0]
        # # 时间
        #
        # date = tags['Image DateTime'].values[0]
        # lat = dj_data_dict.get('GpsLatitude')
        # lon = dj_data_dict.get('GpsLongitude')
        # absoluteAltitude = dj_data_dict.get('AbsoluteAltitude')
        # relativeAltitude = dj_data_dict.get('RelativeAltitude')
        # uavYaw = dj_data_dict.get('FlightYawDegree')
        # uavRoll = dj_data_dict.get('FlightRollDegree')
        # uavPitch = dj_data_dict.get('FlightPitchDegree')
        # cameraYaw = dj_data_dict.get('GimbalYawDegree')
        # cameraRoll = dj_data_dict.get('GimbalRollDegree')
        # cameraPitch = dj_data_dict.get('GimbalPitchDegree')

        # 创建一个数据字典
        data_row = {
            'num': num,
            "camera":tags['Image Model'].values,
            'filename':filename,
            'date': tags['Image DateTime'].values,
            'lat': dj_data_dict.get('GpsLatitude').strip('+'),
            'lon': dj_data_dict.get('GpsLongitude').strip('+'),
            'absoluteAltitude':dj_data_dict.get('AbsoluteAltitude').strip('+'),
            'relativeAltitude': dj_data_dict.get('RelativeAltitude').strip('+'),
            'uavYaw': dj_data_dict.get('FlightYawDegree').strip('+'),
            'uavRoll': dj_data_dict.get('FlightRollDegree').strip('+'),
            'uavPitch': dj_data_dict.get('FlightPitchDegree').strip('+'),
            'cameraYaw':dj_data_dict.get('GimbalYawDegree').strip('+'),
            'cameraRoll': dj_data_dict.get('GimbalRollDegree').strip('+'),
            'cameraPitch': dj_data_dict.get('GimbalPitchDegree').strip('+'),
            'focal':float(tags['EXIF FocalLength'].values[0])
        }
        # 如果文件不存在，写入标题行
        if not os.path.exists(csv_file_path):
            header = list(data_row.keys())
            # 创建一个 DataFrame 并写入标题
            pd.DataFrame(columns=header).to_csv(csv_file_path, index=False)

        # 将数据追加到 CSV 文件
        dataframe = pd.DataFrame([data_row])
        dataframe.to_csv(csv_file_path, mode='a', header=False, index=False)

        num+=1

        # fieldnames = ["num", 'filename', 'date', 'lat', 'lon', 'absoluteAltitude',
        #               'relativeAltitude', 'uavYaw', 'uavRoll', 'uavPitch', 'cameraYaw',
        #               'cameraRoll', 'cameraPitch']
    return


def rename_image(folder_path):
    """ 将文件夹下所有图像按照11_0001的前缀依次重命名"""
    # 获取文件夹内所有文件
    files = os.listdir(folder_path)

    # 初始化计数器
    counter = 1

    # 按照指定的格式重命名文件
    for file_name in files:
        # 过滤出图像文件（根据需要可以修改扩展名）
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 生成新的文件名
            new_name = f"24_{counter:04d}{os.path.splitext(file_name)[1]}"
            # 完整路径
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(folder_path, new_name)
            # 重命名
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} to {new_file_path}')
            # 计数器自增
            counter += 1


if __name__ == "__main__":
    # rename_image(r"I:\datasets\ComplexUAV\24\drone")
    write_csv(r"I:\datasets\ComplexUAV",scene=24)
