import argparse
import os
import time
import json
import shutil
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from pydicom import Dataset as DICOMDataset
from scipy import io
import timm
from pytorch_msssim import ssim

EPOCHS = 300


class TrainDataset(Dataset):
    def __init__(self, input, target, transform = None):
        self.transform = transform
        input_2 = np.array([input +"/"+ x  for x in os.listdir(input)])
        target_forward = np.array([target +"/"+ x  for x in os.listdir(target)])
        
        assert len(input_2) == len(target_forward)
        
        input_2.sort()
        target_forward.sort()

        self.data = {'input': input_2, 'target': target_forward}
            
    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, idx):
        input_path = self.data['input'][idx]
        target_path = self.data['target'][idx]
        
        input_img = io.loadmat(input_path)['img'].astype('float32')
        target_img = io.loadmat(target_path)['img'].astype('float32')
        
        input_img = np.repeat(input_img[:,:,np.newaxis], 3, axis=2)
        input_img = torch.from_numpy(input_img).permute(2, 0, 1).float()
        
        # 3. 处理 target_img 为单通道 (1, H, W)
        target_img = torch.from_numpy(target_img).unsqueeze(0).float()  # (1, H, W)

        # 4. 应用 transforms（如果有的话）
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img.repeat(3, 1, 1))[0:1, :, :]  # 保持单通道

        return {"input_img": input_img, "target_img": target_img}


class SwinPETModel(nn.Module):
    def __init__(self):
        super(SwinPETModel, self).__init__()
        self.model = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=True, patch_size=1,
            pretrained_cfg_overlay=dict(file="./swin_base_patch4_window7_224_22kto1k.pth")
        )
        self.model.head = nn.Identity()  # 移除分类头

        self.upsample_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64 * 8, kernel_size=3, padding=1),  # 修改通道数为 8×8=64
            nn.PixelShuffle(8),  # 8 倍上采样
            nn.Conv2d(8, 1, kernel_size=3, padding=1)  # 保证输入通道数为 8
        )


    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        x = self.upsample_conv(x)
        return x

class MSESSIMLoss(nn.Module):
    def __init__(self, alpha=0.8, ssim_window_size=11):
        """组合损失函数: MSE + SSIM"""
        super(MSESSIMLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim_window_size = ssim_window_size

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        ssim_loss = 1 - ssim(output, target, data_range=1, size_average=True, win_size=self.ssim_window_size)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--output", type=str, default="../models/model_default", help="Checkpoint save path.")
    parser.add_argument("--input", type=str, default="../mat/NAC_train_transformer", help="Input images path.")
    parser.add_argument("--target", type=str, default="../mat/CT_train_transformer", help="Target images path.")
    parser.add_argument("--resume", action="store_true", help="Resume training.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def setup():
    dist.init_process_group("nccl", init_method="env://")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def main(world_size, args):
    setup()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % world_size)
    device = torch.device(rank % world_size)
    
    # 载入数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Swin Transformer 默认输入大小为 224x224
    ])
    dataset = TrainDataset(input=args.input, target=args.target, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1, sampler=sampler, pin_memory=True)

    model = SwinPETModel().to(device)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if args.resume and rank == 0:
        checkpoint_path = os.path.join(args.output, "checkpoint/latest.pth")
        if os.path.exists(checkpoint_path):
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"[INFO] Loaded checkpoint from {checkpoint_path}")

    criterion = MSESSIMLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.5)
    
    for epoch in range(EPOCHS):
        epoch_time = time.time()
        net.train()
        dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0
        num_batches = len(dataloader)
        for batch in dataloader:
            input_img = batch["input_img"].to(device)
            target_img = batch["target_img"].to(device)

            optimizer.zero_grad()
            output = model(input_img)
            loss = criterion(output, target_img)
           
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0 and rank == 0:
            state = net.state_dict()
            path1 = os.path.join(args.output, "checkpoint/%04d.pth"%epoch)
            torch.save(state, path1)
            shutil.copy2(path1, os.path.join(args.output, "checkpoint/latest.pth"))

        if rank == 0:
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch} | time: {(time.time() - epoch_time):.2f} | loss: {(epoch_loss / num_batches):.6f} | lr: {(optimizer.param_groups[0]['lr']):.6f}")    
        
        scheduler.step()
    
    if rank == 0:
        state = net.state_dict()
        torch.save(state, os.path.join(args.output, "checkpoint/result.pth"))
        print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train finished.")

        scheduler.step()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    args = parse_arguments()
    main(world_size, args)
