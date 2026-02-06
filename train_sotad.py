import argparse
import os
import glob
import timeit
import copy
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from models.sotad import Generator, get_discriminator
from torchvision import transforms
from data.accident_sotad import GANData
from torch.utils.data import DataLoader
from models.evluations import eval_fun
import models.helper as hf
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import models.loss_F as loss_F
import torch.distributed as dist
from models.sotad import build_extracotr
from utils.utils import yaml_to_dict
from log.logger import Logger

def reduce_tensor(tensor:torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt,op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", dest="batch_size", type=int, required=True)
parser.add_argument("--num_works", dest="num_works", type=int, required=True)
parser.add_argument("--epochs", dest="epochs", type=int, required=True)
parser.add_argument("--val_ep", dest="val_ep", type=int, required=True)
parser.add_argument("--local-rank", "--local_rank", dest="local_rank", default=-1, type=int)
parser.add_argument("--output_dir", dest="output_dir", default="train_sotad", type=str)
parser.add_argument("--sample_space", dest="sample_space", type=int, default=9)
parser.add_argument("--dist", dest="dist", action="store_true", help="enable distributed training")

# config
args = parser.parse_args()
config = yaml_to_dict('configs/train_accident_sotad.yaml')
can_log = config.get('LOGGING', True)
output_dir =f'outputs/{args.output_dir}'
os.makedirs(output_dir, exist_ok=True)
use_ddp = args.dist

# logging
train_logger = Logger(logdir=os.path.join(output_dir, "log"), only_main=True)
train_logger.show(head="Configs:", log=config)
train_logger.write(log=config, filename="config.yaml", mode="w")
local_rank = args.local_rank

# DDP：DDP backend initialization
if use_ddp:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl is the fastest and most recommended backend on GPU devices
    distributed = True
else:
    distributed = False
    local_rank = 0

# Log record preparation
save_dir_root = output_dir
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
resume_epoch = 0  # Default is 0, change if want to resume
runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

batch_size = args.batch_size
num_workers = args.num_works
epochs = args.epochs
val_epoch = args.val_ep
lr_d = config.get('LR_D', 0.0002)
lr_g = config.get('LR_G', 0.002)

root = config.get('DATASET_ROOT', "datasets/so-tad") # Data set root path needs to be configured

transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Gan datasets preparing
# logger.info("Gan datasets preparing!")

    # print("Gan datasets preparing!")
train_dataset = GANData(root, mode="train", sample_space=args.sample_space, transform=transform)
test_dataset = GANData(root, mode="test", sample_space=args.sample_space, transform=transform)

if distributed:
    train_gan_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_gan_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,shuffle=False)
else:
    train_gan_sampler = None
    test_gan_sampler = None

train_loader = DataLoader(train_dataset,sampler=train_gan_sampler,
                              batch_size=batch_size,
                              shuffle=not distributed,
                              num_workers=num_workers,
                              drop_last=True)

test_loader = DataLoader(test_dataset,sampler=test_gan_sampler,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=True)

extractor = build_extracotr(config).to(local_rank)
# Instantiate the generator and initialize the weights
netG = Generator(16, 64, 16).cuda(local_rank)
netG.apply(hf.weights_init)

train_logger.show(head=f"[INFO] current netD: {config['NETD_VERSION']}")
Discriminator = get_discriminator(config)
netD = Discriminator(16, 16, 16).cuda(local_rank)
netD.apply(hf.weights_init)

if distributed:
    netG = DistributedDataParallel(netG, device_ids=[local_rank], output_device=local_rank)
    netD = DistributedDataParallel(netD, device_ids=[local_rank], output_device=local_rank)

# loss function
criterion = nn.BCELoss()
# Real label and fake label
real_label = config.get('REAL_LABEL', 1.0)
fake_label = config.get('FAKE_LABEL', 0.0)

# Create optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))

# training
img_list = []
G_losses = []
D_losses = []
iters = 0
# Initialize result container
accident_dic = hf.get_key_frames(root)
test_res_dic = hf.init_res_dic(root=root, mode="test")
test_res_dic1 = hf.init_res_dic(root=root, mode="test")
train_res_dic = hf.init_res_dic(root=root, mode="train")
# vae_net.eval()

best_auc = 0
iter_gan = 0

total_len = len(train_loader)

for epoch in range(epochs):
    start_time = timeit.default_timer()
    tmp_train_dic = copy.deepcopy(train_res_dic)
    G_losses.clear()
    D_losses.clear()
    # for i, data in enumerate(tqdm(train_loader, leave=False)):
    for i, data in enumerate(train_loader):
        
        '''
        (1) UpdateD: maximize log(D(x)) + log(1 - D(G(z)))
        '''
        # Batch training using real labels
        netD.zero_grad()
        real_cpu = data[0].to(local_rank)

        t0 = torch.squeeze(real_cpu[:, :, 0:1, :, :], 2)
        t1 = torch.squeeze(real_cpu[:, :, 1:2, :, :], 2)
        t2 = torch.squeeze(real_cpu[:, :, 2:3, :, :], 2)
        t3 = torch.squeeze(real_cpu[:, :, -1:, :, :], 2)
        with torch.no_grad():
            # HND extractor features
            # shape: [B, 256, 77, 57]
            mu0 = extractor(t0)
            mu1 = extractor(t1)
            mu2 = extractor(t2)
            mu3 = extractor(t3)
            
            d01 = nn.functional.normalize(mu1 - mu0).to(local_rank)
            d12 = nn.functional.normalize(mu2 - mu1).to(local_rank)

        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float).to(local_rank)
        output = netD(mu0, mu1, mu2, mu3).view(-1)
        errD_real = criterion(output, label).to(local_rank)
        errD_real.backward()
        D_x = output.mean().item()

        # Batch training using false labels
        noise = torch.autograd.Variable(
            torch.Tensor(np.random.normal(0, 1, (b_size, mu0.size(1), mu0.size(2), mu0.size(3))))).to(local_rank)
        noise1 = torch.autograd.Variable(
            torch.Tensor(np.random.normal(0, 1, (b_size, mu0.size(1), mu0.size(2), mu0.size(3))))).to(local_rank)
        noise2 = torch.autograd.Variable(
            torch.Tensor(np.random.normal(0, 1, (b_size, mu0.size(1), mu0.size(2), mu0.size(3))))).to(local_rank)
        fake = netG(mu0 + noise1, mu1 + noise2, mu2 + noise, d01, d12).to(local_rank)

        label.fill_(fake_label)
        output = netD(mu0, mu1, mu2, fake.detach()).view(-1)
        errD_fake = criterion(output, label).to(local_rank)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizerD.step()


        ############################
        # (2) UpdateG: maximize log(D(G(z)))
        ###########################
        for j in range(1):
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(mu0, mu1, mu2, fake).view(-1)

            errG = criterion(output, label)
            errCOS = loss_F.cosine_similarity(fake, mu3)
            errGDL = loss_F.gdl_loss(fake, mu3,local_rank)
            errINT = loss_F.intensity_loss(fake, mu3)
            # # errG = -torch.mean(output)
            # errG = (errG + errCOS + errGDL + errINT).to(local_rank)
            errG = (errG + errCOS).to(local_rank)

            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
        # Output training status
        if i % 50 == 0:
            
            d_g_z = D_G_z1 / D_G_z2
            log_txt = f'Loss_D: {errD.item():.3f}, LossG: {errG.item():.3f}, D(x): {D_x:.3f}, D(G(z)): {d_g_z:.3f}'
            train_logger.show(head=f"[Epoch {epoch}, {i}/{total_len}]", log=log_txt)
            train_logger.write(head=f"[Epoch {epoch}, {i}/{total_len}]", log=log_txt)
        # Save the loss of each round
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
    av_g_loss = sum(G_losses) / len(G_losses)
    av_d_loss = sum(D_losses) / len(D_losses)
    stop_time = timeit.default_timer()
    if can_log:
        
        log_txt=f'DLoss: {av_d_loss:.3f}, GLoss: {av_g_loss:.3f}'
        train_logger.show(head=f"[Epoch {epoch}] ", log=log_txt)
        train_logger.write(head=f"[Epoch {epoch}] ", log=log_txt)
        execution_time = (stop_time - start_time) / 60
        train_logger.show(head=f"Execution time: {execution_time:.3f} min")
        train_logger.write(head=f"Execution time: {execution_time:.3f} min")

    # test
    if epoch % val_epoch == 0:
        if can_log:
            # logger.info("GAN model start testing!")
            
            # print("GAN model start testing!")
            train_logger.show(head="GAN model start testing")
        start_time = timeit.default_timer()
        tmp_test_dic_psnr = copy.deepcopy(test_res_dic1)

        netG.eval()
        netD.eval()

        for it, data in enumerate(tqdm(test_loader, leave=False)):
            netD.zero_grad()
            netG.zero_grad()

            real_cpu = data[0].to(local_rank)  # (B, C, T, H, W)

            # 拆 t0~t3
            t0 = torch.squeeze(real_cpu[:, :, 0:1, :, :], 2)
            t1 = torch.squeeze(real_cpu[:, :, 1:2, :, :], 2)
            t2 = torch.squeeze(real_cpu[:, :, 2:3, :, :], 2)
            t3 = torch.squeeze(real_cpu[:, :, -1:, :, :], 2)

            with torch.no_grad():
                # -------- 提取 HND 特征 --------
                mu0 = extractor(t0)   # [B,256,77,57]
                mu1 = extractor(t1)
                mu2 = extractor(t2)
                mu3 = extractor(t3)

                d01 = nn.functional.normalize(mu1 - mu0, dim=1)
                d12 = nn.functional.normalize(mu2 - mu1, dim=1)

                # 生成预测特征 fake_mu3
                fake = netG(mu0, mu1, mu2, d01, d12).detach()  # [B,256,77,57]

                # -------- 计算每个 sample 的余弦距离分数 --------
                batch_size = fake.shape[0]
                p = fake.view(batch_size, -1)   # [B, N]
                x = mu3.view(batch_size, -1)    # [B, N]
                cos = 1 - F.cosine_similarity(p, x, dim=1)  # [B]
                res_cos = cos.detach()  # [B], float32, on local_rank

                if distributed:
                    # ================== 多卡收集 ==================
                    world_size = dist.get_world_size()

                    # 视频 index / 结束帧 index 搬到 GPU，保证 device 一致
                    videos = data[1].to(local_rank).long()  # [B]
                    frames = data[3].to(local_rank).long()  # [B]

                    # 准备好 all_gather 的 buffer，dtype/shape/device 与输入一致
                    scores_gather = [torch.zeros_like(res_cos) for _ in range(world_size)]
                    videos_gather = [torch.zeros_like(videos)  for _ in range(world_size)]
                    frames_gather = [torch.zeros_like(frames)  for _ in range(world_size)]

                    dist.all_gather(scores_gather, res_cos)
                    dist.all_gather(videos_gather, videos)
                    dist.all_gather(frames_gather, frames)

                    # 展平
                    score_list = torch.cat(scores_gather, dim=0).cpu().numpy()
                    video_list = torch.cat(videos_gather, dim=0).cpu().numpy()
                    frame_list = torch.cat(frames_gather, dim=0).cpu().numpy()

                else:
                    # ================== 单卡 / debug 模式 ==================
                    score_list = res_cos.cpu().numpy()
                    video_list = data[1].cpu().numpy()
                    frame_list = data[3].cpu().numpy()

                # 写入到结果字典中
                hf.preds_to_dic(tmp_test_dic_psnr, score_list, video_list, frame_list)

        nrmse_10, F1_10, AUC_10 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.1)
        nrmse_20, F1_20, AUC_20 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.2)
        nrmse_30, F1_30, AUC_30 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.3)
        stop_time = timeit.default_timer()

        if can_log:
            train_logger.show(head=f"[epoch: {epoch}] GAN parameters saveing!")
            os.makedirs(f'{output_dir}/parms', exist_ok=True)
            torch.save(netD.state_dict(), f'{output_dir}/parms/{epoch}_netD.pt')
            torch.save(netG.state_dict(), f'{output_dir}/parms/{epoch}_netG.pt')
            
            train_logger.show(head=f"[GAN Val] Epoch: {epoch}/{epochs-1}, threshold: 0.1, nrmse: {nrmse_10}, F1: {F1_10}, AUC: {AUC_10}")
            train_logger.write(head=f"[GAN Val] Epoch: {epoch}/{epochs-1}, threshold: 0.1, nrmse: {nrmse_10}, F1: {F1_10}, AUC: {AUC_10}")

            train_logger.show(head=f"[GAN Val] Epoch: {epoch}/{epochs-1}, threshold: 0.2, nrmse: {nrmse_20}, F1: {F1_20}, AUC: {AUC_20}")
            train_logger.write(head=f"[GAN Val] Epoch: {epoch}/{epochs-1}, threshold: 0.2, nrmse: {nrmse_20}, F1: {F1_20}, AUC: {AUC_20}")

            train_logger.show(head=f"[GAN Val] Epoch: {epoch}/{epochs-1}, threshold: 0.3, nrmse: {nrmse_30}, F1: {F1_30}, AUC: {AUC_30}")
            train_logger.write(head=f"[GAN Val] Epoch: {epoch}/{epochs-1}, threshold: 0.3, nrmse: {nrmse_30}, F1: {F1_30}, AUC: {AUC_30}")
            
        # This ensures that other processes can only read the model after the model is saved.
        if distributed:
            dist.barrier()
        netG.train()
        netD.train()