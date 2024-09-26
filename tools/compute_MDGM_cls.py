import re

import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model', default='vmamba_s', type=str,
                    help='ViTS-384 | convnext_t | convnext_s | convnext_b | ViTB-384 | resnet50 | vmamba_b | vmamba_s')

parser.add_argument('--data',
                    default='/home/hello/cjl/Sample4Geo-main/result/cls/vmamba_s/visloc/ce/all_datasets_evaluation_results.txt',
                    type=str, help='数据地址')
args = parser.parse_args()

if args.model == "resnet50":
    DenseUAV_AP = 25.24
    SUES_AP = 48.82
    University_AP = 62.91
    ComplexUAV_AP = 36.21
elif args.model == "convnext_t":
    DenseUAV_AP = 68.08
    SUES_AP = 81.26
    University_AP = 81.50
    ComplexUAV_AP = 62.67
elif args.model == "convnext_s":
    DenseUAV_AP = 74.18
    SUES_AP = 84.44
    University_AP = 85.52
    ComplexUAV_AP = 65.58
elif args.model == "convnext_b":
    DenseUAV_AP = 80.26
    SUES_AP = 88.54
    University_AP = 86.88
    ComplexUAV_AP = 66.85
elif args.model == "ViTS-384":
    DenseUAV_AP = 25.34
    SUES_AP = 78.60
    University_AP = 82.74
    ComplexUAV_AP = 63.16
elif args.model == "ViTB-384":
    DenseUAV_AP = 76.30
    SUES_AP = 85.17
    University_AP = 84.36
    ComplexUAV_AP = 76.98
elif args.model == "vmamba_b":
    DenseUAV_AP = 27.71
    SUES_AP = 80.84
    University_AP = 87.16
    ComplexUAV_AP = 65.23
elif args.model == "vmamba_s":
    DenseUAV_AP =28.34
    SUES_AP = 81.08
    University_AP = 85.64
    ComplexUAV_AP = 66.16

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
