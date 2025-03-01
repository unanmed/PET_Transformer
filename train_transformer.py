import os, time
import json
from pydicom import Dataset
from scipy import io
import timm
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_msssim import ssim
from config.config import get_arguments

EPOCHES = 300

def generate_mask(img_height,img_width,radius,center_x,center_y):
    y,x=np.ogrid[0:img_height,0:img_width]
    # circle mask
    mask = (x-center_x)**2+(y-center_y)**2<=radius**2
    return mask

class mriDataset(Dataset):
    def __init__(self, opt, root1, root2, root3, transform = None): 
        self.transform = transform
        self.task = opt.task
        input_2 = np.array([root2 +"/"+ x  for x in os.listdir(root2)])
        target_forward = np.array([root1 +"/"+ x  for x in os.listdir(root1)])
        input_3 = np.array([root3 +"/"+ x  for x in os.listdir(root3)])
        
        assert len(input_2) == len(target_forward) == len(input_3)
        
        input_2.sort()
        input_3.sort()
        target_forward.sort()

        self.data = {'input_2':input_2, 'target_forward':target_forward,'input_3':input_3}
            
    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)

    def __len__(self):
        return len(self.data['target_forward'])

    def __getitem__(self, idx):        
        input_2_path = self.data['input_2'][idx]
        target_forward_path = self.data['target_forward'][idx]
        input_3_path = self.data['input_3'][idx]
        
        assert (input_2_path.split('/')[-1]) == (target_forward_path.split('/')[-1]) == (input_3_path.split('/')[-1])
        
        input_2_data = io.loadmat(input_2_path)['img'].astype('float32')
        target_forward_data = io.loadmat(target_forward_path)['img'].astype('float32')
        
        print(input_2_data.shape)
        
        mask = generate_mask(256, 256, 128, 128, 128)
            
        input_2_data = input_2_data * mask
        target_forward_data = target_forward_data * mask
        
        h,w = input_2_data.shape

        target_forward_img = np.expand_dims(target_forward_data, 2) 
        target_forward_img = np.concatenate((target_forward_img,target_forward_img,target_forward_img),axis=2)

        input_2_img = np.expand_dims(input_2_data, 2) 
        input_img = np.zeros((h,w,3))
        
        if self.task== '1to1':
            input_img[:,:,0] = input_2_img[:,:,0]
            input_img[:,:,1] = input_2_img[:,:,0]
            input_img[:,:,2] = input_2_img[:,:,0]

        input_target_img = input_img.copy()

        input_img = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).float()
        target_forward_img = torch.from_numpy(target_forward_img).permute(2, 0, 1).unsqueeze(0).float()
        input_target_img = torch.from_numpy(input_target_img).permute(2, 0, 1).unsqueeze(0).float()
        
        if self.transform:
            input_img = self.transform(input_img)
            target_forward_img = self.transform(target_forward_img)
            input_target_img = self.transform(input_target_img)
            
        # print(input_img.shape, input_2_path)

        sample = {
            'input_img': input_img, 
            'target_forward_img': target_forward_img, 
            'input_target_img': input_target_img,
            'input2_name': input_2_path.split("/")[-1].split(".")[0],
            'input3_name': input_3_path.split("/")[-1].split(".")[0],
            'target_forward_name': target_forward_path.split("/")[-1].split(".")[0]
        }
        return sample

class SwinPETModel(nn.Module):
    def __init__(self):
        super(SwinPETModel, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, pretrained_cfg_overlay=dict(file="./swin_base_patch4_window7_224_22kto1k.pth"))
        
        # Swin Transformer 的 head 修改为像素级输出
        self.model.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1)  # 输出单通道 (灰度图)
        )

    def forward(self, x):
        print(x.shape)
        return self.model(x)
    
# 混合损失函数 (L1 Loss + SSIM Loss)
class MixLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(MixLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1, size_average=True)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
    
def parse_arguments():
    parser = get_arguments()
    parser.add_argument("--out_path", type=str, default="./results/", help="Path to save checkpoint. ")
    parser.add_argument("--root1", type=str, default="./data/T2", help="Output images. ")
    parser.add_argument("--root2", type=str, default="./data/T1", help="Input images. ")
    parser.add_argument("--root3", type=str, default="./data/PD", help="Another input images. ")
    parser.add_argument("--resume", dest='resume', action='store_true',  help="Resume training. ")
    parser.add_argument("--loss", type=str, default="L2", choices=["L1", "L2"], help="Choose which loss function to use. ")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args
    
def init_status(args):
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.out_path+"%s"%args.task, exist_ok=True)
    os.makedirs(args.out_path+"%s/checkpoint"%args.task, exist_ok=True)
    with open(args.out_path+"%s/commandline_args.yaml"%args.task , 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
def setup():
    """初始化分布式训练环境"""
    dist.init_process_group("nccl", init_method="env://")  # NCCL 后端（最快）

def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main(world_size, args):
    # 初始化 DDP
    setup()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % world_size)
    device = torch.device(rank % world_size)
    
    # 载入数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Swin Transformer 默认输入大小为 224x224
    ])
    dataset = mriDataset(opt=args, root1=args.root1, root2=args.root2, root3=args.root3, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=2, num_workers=4, drop_last=True,
        prefetch_factor=2, pin_memory=True, sampler=sampler
    )
    
    # 定义模型
    model = SwinPETModel().to(device)
    net = DDP(model)

    if args.resume and rank == 0:
        checkpoint_path = f"{args.out_path}/{args.task}/checkpoint/latest.pth"
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] loaded " + args.out_path+"%s/checkpoint/latest.pth"%args.task)
        
    # 优化器和调度器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250], gamma=0.5)

    # 数据记录
    # writer = SummaryWriter(args.out_path+"%s"%args.task)
    
    # 初始化训练
    step = 0
    loss_all = np.zeros((300), dtype='float')
    num_batches = len(dataloader)
    
    print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start to train")

    for epoch in range(EPOCHES):
        epoch_time = time.time()
        net.train()
        loss_this_time = 0
        dataloader.sampler.set_epoch(epoch)
        for i_batch, sample_batched in enumerate(dataloader):
            step_time = time.time()
            
            input = sample_batched['input_img'].to(device, non_blocking=True)
            target_forward = sample_batched['target_forward_img'].to(device, non_blocking=True)
            # input_target = sample_batched['input_target_img'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(input)
            
            if rank == 0:
                print('epoch: ' + str(epoch) + ' iter: ' + str(i_batch) +' loss: ' + str(loss.item()))
            
            loss = criterion(output, target_forward)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            loss_this_time = loss_this_time + loss
            step += 1
            
        loss_this_time = loss_this_time / num_batches
        loss_all[epoch] = loss_this_time
        
        torch.save(net.state_dict(), args.out_path+"%s/checkpoint/latest.pth"%args.task)
        if epoch % 1 == 0 and rank == 0:
            # os.makedirs(args.out_path+"%s/checkpoint/%04d"%(args.task,epoch), exist_ok=True)
            torch.save(net.state_dict(), args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully saved "+args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))
            
        if epoch % 10 == 0 and rank == 0:
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]task: %s Epoch: %d Step: %d || loss: %.5f || lr: %f time: %f"%(
                args.task, epoch, step, loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time() - step_time
            ))

        scheduler.step()   
        
        if rank == 0:
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch time: ", time.time()-epoch_time, "task: ", args.task)    
    
    if rank == 0:
        print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train finished.")
    cleanup()

if __name__ == '__main__':
    try:
        world_size = torch.cuda.device_count()
        args = parse_arguments()
        init_status(args)
        torch.set_num_threads(4)
        main(world_size, args)
    finally:
        cleanup()
