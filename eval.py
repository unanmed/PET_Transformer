import argparse
import os
from scipy import io
import torch
import timm
import numpy as np
import cv2
import torch.nn as nn
from torchvision import transforms
from train import SwinPETModel, MSESSIMLoss
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--output", type=str, default="../results/model_default", help="Path to save checkpoint.")
    parser.add_argument("--input", type=str, default="../mat/NAC_test", help="Input images.")
    parser.add_argument("--target", type=str, default="../mat/CT_test", help="Target images.")
    parser.add_argument("--model", type=str, default="../models/model_default/checkpoint/latest.pth")
    args = parser.parse_args()
    return args

def init_status(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "predict"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "target"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "origin"), exist_ok=True)

def eval(args):
    # 🧠 2. 加载训练好的模型权重
    model = SwinPETModel()
    # 加载模型权重字典
    checkpoint = torch.load(args.model, map_location='cpu')
    criterion = MSESSIMLoss()

    # 处理 state_dict，移除 'module.' 前缀
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
        else:
            new_state_dict[k] = v

    # 将新 state_dict 加载到模型中
    model.load_state_dict(new_state_dict)
    model.eval()  # 设为推理模式
    
    # 🌈 3. 定义图像预处理方法 (与训练时保持一致)
    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])
    
    input_folder = args.input  # 存放 .mat 文件的文件夹
    target_folder = args.target  # 期望输出图片
    output_folder = args.output  # 预测输出图片的保存文件夹
    
    loss_total = 0
    
    for file_name in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        
        # 读取 .mat 文件中的图像数据 (假设 img 字段包含 256x256 的灰度图，范围为 0-1)
        input_img = io.loadmat(input_path)['img'].astype('float32')
        input_img = np.repeat(input_img[:,:,np.newaxis], 3, axis=2)
        input_img = torch.from_numpy(input_img).permute(2, 0, 1).float()
        input_img = transform(input_img).unsqueeze(0)
        
        target_img = io.loadmat(target_path)['img'].astype('float32')
        target_img = torch.from_numpy(target_img).unsqueeze(0).float()
        target_img = transform(target_img.repeat(3, 1, 1))[0:1, :, :].unsqueeze(0)
        
        # 进行推理
        with torch.no_grad():
            output = model(input_img)
        
        loss = criterion(output, target_img)
        loss_total += loss
        
        # 处理输出 (转换为图像)
        output = output.squeeze(0).squeeze(0).numpy()  # 形状 (224, 224)
        output = (output - output.min()) / (output.max() - output.min())  # 归一化到 0-1
        output = (output * 255).astype(np.uint8)
        
        # 保存输出图像
        output_path = os.path.join(output_folder, "predict", f"{os.path.splitext(file_name)[0]}.png")
        cv2.imwrite(output_path, output)
        
    print(f"All precessed loss: {loss_total / len(os.listdir(input_folder))}")

if __name__ == "__main__":
    args = parse_arguments()
    init_status(args)
    eval(args)