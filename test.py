import torch

# 加载第一个文件
file1 = "/home/data/luoxi/SSR/ssr/dino/dino_features.pt"
# 加载第一个文件并提取 features
data1 = torch.load(file1)
if 'features' in data1:
    features1 = data1['features']
    print(f"Features of {file1}:")
    print(features1)
else:
    print(f"'features' not found in {file1}")

# # 加载第二个文件
# file2 = "/home/data/luoxi/SSR/ssr/dino/dino_features.pt"
# data2 = torch.load(file2)
# print(f"Contents of {file2}:")
# print(data2)
