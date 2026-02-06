# @Author       : zetton
# @Date         : 2026
import torch

from utils.utils import distributed_rank
from .hncd import build as build_hncd


def build_model(config: dict):
    model = build_hncd(config=config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model
