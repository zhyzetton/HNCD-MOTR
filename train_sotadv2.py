import argparse
import os
import glob
import timeit
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.sotad import Generator, get_discriminator, build_extracotr
from data.accident_sotad import GANData
from models.evluations import eval_fun
import models.helper as hf
import models.loss_F as loss_F
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

args = parser.parse_args()
config = yaml_to_dict('configs/train_accident_sotad.yaml')
can_log = config.get('LOGGING', True)
output_dir = f'outputs/{args.output_dir}'
os.makedirs(output_dir, exist_ok=True)
use_ddp = args.dist

# logging
train_logger = Logger(logdir=os.path.join(output_dir, "log"), only_main=True)
train_logger.show(head="Configs:", log=config)
train_logger.write(log=config, filename="config.yaml", mode="w")
local_rank = args.local_rank

if use_ddp:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    distributed = True
else:
    distributed = False
    local_rank = 0

batch_size = args.batch_size
num_workers = args.num_works
epochs = args.epochs
val_epoch = args.val_ep
lr_d = config.get('LR_D', 0.0002)
lr_g = config.get('LR_G', 0.002)

root = config.get('DATASET_ROOT', "datasets/so-tad")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = GANData(root, mode="train", sample_space=args.sample_space, transform=transform)
test_dataset = GANData(root, mode="test", sample_space=args.sample_space, transform=transform)

if distributed:
    train_gan_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_gan_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
else:
    train_gan_sampler = None
    test_gan_sampler = None

train_loader = DataLoader(train_dataset, sampler=train_gan_sampler, batch_size=batch_size,
                          shuffle=not distributed, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_dataset, sampler=test_gan_sampler, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers, drop_last=True)

extractor = build_extracotr(config).to(local_rank)
netG = Generator(16, 64, 16).cuda(local_rank)
netG.apply(hf.weights_init)

train_logger.show(head=f"[INFO] current netD: {config['NETD_VERSION']}")
Discriminator = get_discriminator(config)
netD = Discriminator(16, 16, 16).cuda(local_rank)
netD.apply(hf.weights_init)

if distributed:
    netG = DistributedDataParallel(netG, device_ids=[local_rank], output_device=local_rank)
    netD = DistributedDataParallel(netD, device_ids=[local_rank], output_device=local_rank)
    extractor = DistributedDataParallel(extractor, device_ids=[local_rank], output_device=local_rank)

criterion = nn.BCELoss()
real_label = config.get('REAL_LABEL', 1.0)
fake_label = config.get('FAKE_LABEL', 0.0)

optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))
# 将 Adapter 参数加入优化器
trainable_extractor_params = [p for p in extractor.parameters() if p.requires_grad]
optimizerG = optim.Adam(list(netG.parameters()) + trainable_extractor_params, lr=lr_g, betas=(0.5, 0.999))

accident_dic = hf.get_key_frames(root)
test_res_dic1 = hf.init_res_dic(root=root, mode="test")
train_res_dic = hf.init_res_dic(root=root, mode="train")

total_len = len(train_loader)

for epoch in range(epochs):
    start_time = timeit.default_timer()
    G_losses, D_losses = [], []
    
    if distributed:
        train_gan_sampler.set_epoch(epoch)

    for i, data in enumerate(train_loader):
        netD.zero_grad()
        real_cpu = data[0].to(local_rank)

        t0 = torch.squeeze(real_cpu[:, :, 0:1, :, :], 2)
        t1 = torch.squeeze(real_cpu[:, :, 1:2, :, :], 2)
        t2 = torch.squeeze(real_cpu[:, :, 2:3, :, :], 2)
        t3 = torch.squeeze(real_cpu[:, :, -1:, :, :], 2)

        # 提取特征 (Adapter 可学习)
        mu0 = extractor(t0)
        mu1 = extractor(t1)
        mu2 = extractor(t2)
        mu3 = extractor(t3)
            
        d01 = nn.functional.normalize(mu1 - mu0, dim=1)
        d12 = nn.functional.normalize(mu2 - mu1, dim=1)

        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float).to(local_rank)
        
        # D Real
        output = netD(mu0.detach(), mu1.detach(), mu2.detach(), mu3.detach()).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # 生成噪声与假样本 (恢复为 standard normal)
        noise1 = torch.randn_like(mu0)
        noise2 = torch.randn_like(mu0)
        noise = torch.randn_like(mu0)
        fake = netG(mu0 + noise1, mu1 + noise2, mu2 + noise, d01, d12)

        # D Fake
        label.fill_(fake_label)
        output = netD(mu0.detach(), mu1.detach(), mu2.detach(), fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update G & Adapter
        optimizerG.zero_grad()
        label.fill_(real_label)
        output = netD(mu0, mu1, mu2, fake).view(-1)

        errG_adv = criterion(output, label)
        errCOS = loss_F.cosine_similarity(fake, mu3)
        errG = (errG_adv + errCOS).to(local_rank)

        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            d_g_z = D_G_z1 / (D_G_z2 + 1e-8)
            log_txt = f'Loss_D: {errD.item():.3f}, LossG: {errG.item():.3f}, D(x): {D_x:.3f}, D(G(z)): {d_g_z:.3f}'
            train_logger.show(head=f"[Epoch {epoch}, {i}/{total_len}]", log=log_txt)
            train_logger.write(head=f"[Epoch {epoch}, {i}/{total_len}]", log=log_txt) # 恢复写日志
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    
    # Epoch 结束总结
    av_g_loss = sum(G_losses) / len(G_losses)
    av_d_loss = sum(D_losses) / len(D_losses)
    stop_time = timeit.default_timer()
    if can_log:
        log_txt = f'DLoss: {av_d_loss:.3f}, GLoss: {av_g_loss:.3f}'
        train_logger.show(head=f"[Epoch {epoch}] ", log=log_txt)
        train_logger.write(head=f"[Epoch {epoch}] ", log=log_txt) # 恢复写日志
        execution_time = (stop_time - start_time) / 60
        train_logger.show(head=f"Execution time: {execution_time:.3f} min")
        train_logger.write(head=f"Execution time: {execution_time:.3f} min") # 恢复写日志

    # --- Validation ---
    if epoch % val_epoch == 0:
        train_logger.show(head="GAN model start testing")
        tmp_test_dic_psnr = copy.deepcopy(test_res_dic1)

        netG.eval()
        netD.eval()
        extractor.eval()

        for it, data in enumerate(tqdm(test_loader, leave=False)):
            real_cpu = data[0].to(local_rank)
            t0 = torch.squeeze(real_cpu[:, :, 0:1, :, :], 2)
            t1 = torch.squeeze(real_cpu[:, :, 1:2, :, :], 2)
            t2 = torch.squeeze(real_cpu[:, :, 2:3, :, :], 2)
            t3 = torch.squeeze(real_cpu[:, :, -1:, :, :], 2)

            with torch.no_grad():
                m0, m1, m2, m3 = extractor(t0), extractor(t1), extractor(t2), extractor(t3)
                d01_v = nn.functional.normalize(m1 - m0, dim=1)
                d12_v = nn.functional.normalize(m2 - m1, dim=1)
                fake_v = netG(m0, m1, m2, d01_v, d12_v)

                batch_size_v = fake_v.shape[0]
                p = fake_v.view(batch_size_v, -1)
                x = m3.view(batch_size_v, -1)
                cos = 1 - F.cosine_similarity(p, x, dim=1)
                res_cos = cos.detach()

                if distributed:
                    videos = data[1].to(local_rank).long()
                    frames = data[3].to(local_rank).long()
                    scores_gather = [torch.zeros_like(res_cos) for _ in range(dist.get_world_size())]
                    videos_gather = [torch.zeros_like(videos)  for _ in range(dist.get_world_size())]
                    frames_gather = [torch.zeros_like(frames)  for _ in range(dist.get_world_size())]
                    dist.all_gather(scores_gather, res_cos)
                    dist.all_gather(videos_gather, videos)
                    dist.all_gather(frames_gather, frames)
                    score_list = torch.cat(scores_gather, dim=0).cpu().numpy()
                    video_list = torch.cat(videos_gather, dim=0).cpu().numpy()
                    frame_list = torch.cat(frames_gather, dim=0).cpu().numpy()
                else:
                    score_list, video_list, frame_list = res_cos.cpu().numpy(), data[1].numpy(), data[3].numpy()

                hf.preds_to_dic(tmp_test_dic_psnr, score_list, video_list, frame_list)

        nrmse_10, F1_10, AUC_10 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.1)
        nrmse_20, F1_20, AUC_20 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.2)
        nrmse_30, F1_30, AUC_30 = eval_fun(tmp_test_dic_psnr, accident_dic, threshold=0.3)
        
        if can_log and (not distributed or dist.get_rank() == 0):
            os.makedirs(f'{output_dir}/parms', exist_ok=True)
            def save_m(m, name):
                sd = m.module.state_dict() if hasattr(m, 'module') else m.state_dict()
                torch.save(sd, f'{output_dir}/parms/{epoch}_{name}.pt')
            
            save_m(netD, "netD")
            save_m(netG, "netG")
            save_m(extractor, "extractor")
            
            # 记录详细验证结果到日志
            train_logger.show(head=f"[GAN Val] Epoch: {epoch}, threshold: 0.1, nrmse: {nrmse_10}, F1: {F1_10}, AUC: {AUC_10}")
            train_logger.write(head=f"[GAN Val] Epoch: {epoch}, threshold: 0.1, nrmse: {nrmse_10}, F1: {F1_10}, AUC: {AUC_10}")
            train_logger.write(head=f"[GAN Val] Epoch: {epoch}, threshold: 0.2, nrmse: {nrmse_20}, F1: {F1_20}, AUC: {AUC_20}")
            train_logger.write(head=f"[GAN Val] Epoch: {epoch}, threshold: 0.3, nrmse: {nrmse_30}, F1: {F1_30}, AUC: {AUC_30}")

        if distributed: dist.barrier()
        netG.train()
        netD.train()
        extractor.train()