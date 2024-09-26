import re

import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model', default='vmamba_s', type=str,
                    help='ViTS-384 | convnext_t | convnext_s | convnext_b | ViTB-384 | resnet50 | vmamba_b | vmamba_s')

parser.add_argument('--data',
                    default='/home/hello/cjl/Sample4Geo-main/result/info/vmamba_s/visloc/2024-09-21-13-43-15/all_datasets_evaluation_results.txt',
                    type=str, help='数据地址')
args = parser.parse_args()

if args.model == "resnet50":
    DenseUAV_AP = 66.83
    SUES_AP = 60.54
    University_AP = 80.79
    ComplexUAV_AP = 68.17
elif args.model == "convnext_t":
    DenseUAV_AP = 93.64
    SUES_AP = 86.70
    University_AP = 89.14
    ComplexUAV_AP = 85.65
elif args.model == "convnext_s":
    DenseUAV_AP = 95.19
    SUES_AP = 95.12
    University_AP = 95.80
    ComplexUAV_AP = 90.43
elif args.model == "convnext_b":
    DenseUAV_AP = 95.20
    SUES_AP = 95.22
    University_AP = 96.25
    ComplexUAV_AP = 90.54
elif args.model == "ViTS-384":
    DenseUAV_AP = 82.29
    SUES_AP = 89.86
    University_AP = 89.55
    ComplexUAV_AP = 86.64
elif args.model == "ViTB-384":
    DenseUAV_AP = 88.97
    SUES_AP = 95.65
    University_AP = 93.15
    ComplexUAV_AP = 88.12
elif args.model == "vmamba_b":
    DenseUAV_AP =82.32
    SUES_AP =92.94
    University_AP = 95.42
    ComplexUAV_AP = 81.06
elif args.model == "vmamba_s":
    DenseUAV_AP =82.77
    SUES_AP = 91.12
    University_AP = 94.98
    ComplexUAV_AP =79.91

# 假设数据保存在名为data.txt的文件中
with open(args.data, 'r') as file:
    data = file.read()  # 读取文件内容

# 使用正则表达式匹配每一行和AP值
pattern = r'(\w+):.*?AP:\s+(\S+)'
matches = re.findall(pattern, data)

# 创建字典存储类别名和对应的AP值
ap_values = {name: float(ap) for name, ap in matches}

# 打印结果
for category, ap in ap_values.items():
    print(f"{category}: AP = {ap}")

MDGM = ((ap_values["visloc"] + ap_values["university"]) * (ap_values["denseuav"] + ap_values["sues"])) / (
            (ComplexUAV_AP + University_AP) * (DenseUAV_AP + SUES_AP))
print(f"MDGM = {MDGM}")
