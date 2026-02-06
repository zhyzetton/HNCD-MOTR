import torch
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.utils import inverse_sigmoid


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_cdn_group=3,
    id_noise_ratio=0.3,
    box_noise_scale=0.4,
    cdn_k=5,
    use_tiny_noise=False
):
    """
    Unified Contrastive Denoising (CDN) / Hard Negative Denoising (HND) Group Generator.
    
    Args:
        ...
        cdn_k (int): The 'k' for K-NN sampling, only used if use_hnd_strategies is True.
        use_tiny_noise (bool): If True, enables K-NN sampling and multi-level noise. 
                                   If False, reverts to the original CDN logic (nearest neighbor, no noise on replaced boxes).
    """
    if num_cdn_group <= 0:
        return None, None, None, None

    # --- Section 1, 2, 3: Data Preparation and Masking (This part is identical for both versions) ---
    num_group = num_cdn_group
    num_gts = [len(t["labels"]) for t in targets]
    # Get device from class_embed to ensure consistency, especially when targets have empty tensors on CPU
    device = next(class_embed.parameters()).device
    bs = len(num_gts)

    max_gt_num = max(num_gts) if num_gts else 0
    if max_gt_num == 0:
        # NOTE 构造假数据以确保DDP梯度同步
        padding_idx = num_classes # Use num_classes as padding index for background
        num_fake_queries = 2
        
        fake_query_class = torch.full(
            (bs, num_fake_queries), padding_idx, dtype=torch.long, device=device
        )
        
        fake_query_bbox = torch.tensor(
            [[0.5, 0.5, 0.1, 0.1]], device=device
        ).repeat(bs, num_fake_queries, 1)

        # 生成伪造的 logits 和 bbox_unact
        input_query_logits = class_embed(fake_query_class)
        
        fake_query_bbox = torch.clamp(fake_query_bbox, 1e-6, 1.0 - 1e-6)
        input_query_bbox_unact = inverse_sigmoid(fake_query_bbox)

        # 生成注意力掩码
        tgt_size = num_fake_queries + num_queries
        attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
        attn_mask[:num_fake_queries, num_fake_queries:] = True
        # Optional: attn_mask[num_fake_queries:, :num_fake_queries] = True
        
        # 即使是假数据，也需要一个符合格式的dn_meta
        dn_positive_idx = [torch.empty(0, dtype=torch.long, device=device) for _ in range(bs)]
        dn_meta = {
            "cdn_positive_idx": dn_positive_idx,
            "cdn_num_group": num_group, # Can keep the group num
            "cdn_num_split": [num_fake_queries, num_queries],
        }
        return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta

    input_query_class_list, input_query_bbox_list, pad_gt_mask_list = [], [], []
    for i in range(bs):
        num_gt = num_gts[i]
        cls_tensor = torch.full([max_gt_num], num_classes, dtype=torch.long, device=device)
        box_tensor = torch.zeros([max_gt_num, 4], device=device)
        mask_tensor = torch.zeros([max_gt_num], dtype=torch.bool, device=device)
        if num_gt > 0:
            cls_tensor[:num_gt] = targets[i]["labels"]
            box_tensor[:num_gt] = targets[i]["boxes"]
            mask_tensor[:num_gt] = True
        input_query_class_list.append(cls_tensor)
        input_query_bbox_list.append(box_tensor)
        pad_gt_mask_list.append(mask_tensor)

    input_query_class = torch.stack(input_query_class_list)
    input_query_bbox_padded = torch.stack(input_query_bbox_list)
    pad_gt_mask = torch.stack(pad_gt_mask_list)

    num_denoising = int(max_gt_num * 2 * num_group)
    input_query_class = input_query_class.repeat(1, 2 * num_group)
    input_query_bbox = input_query_bbox_padded.repeat(1, 2 * num_group, 1)
    pad_gt_mask = pad_gt_mask.repeat(1, 2 * num_group)

    base_neg_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    base_neg_mask[:, max_gt_num:] = 1
    neg_mask = base_neg_mask.repeat(1, num_group, 1).bool().squeeze(-1)

    positive_gt_mask = (~neg_mask) & pad_gt_mask
    dn_positive_idx_flat = torch.nonzero(positive_gt_mask)
    dn_positive_idx = []
    for b in range(bs):
        idx_in_batch = dn_positive_idx_flat[dn_positive_idx_flat[:, 0] == b, 1]
        dn_positive_idx.append(idx_in_batch)
    
    # --- Section 4: Hard Negative Generation (Controlled by the new parameter) ---
    replace_mask = (
        (torch.rand(bs, num_denoising, device=device) < id_noise_ratio)
        & neg_mask
        & pad_gt_mask
    )

    for b in range(bs):
        num_gt = num_gts[b]
        if num_gt <= 1:
            continue

        current_gt_boxes = targets[b]["boxes"].to(device)
        gt_centers = current_gt_boxes[:, :2]
        pairwise_dist = torch.cdist(gt_centers, gt_centers, p=2.0)
        pairwise_dist.fill_diagonal_(float("inf"))

        if use_tiny_noise:
            # HND Logic: K-NN Random Sampling
            k_for_topk = min(cdn_k, num_gt - 1)
            if k_for_topk == 0: continue
            _, topk_indices = torch.topk(pairwise_dist, k=k_for_topk, dim=1, largest=False)
            random_choice = torch.randint(k_for_topk, size=(num_gt,), device=device)
            nearest_gt_indices = topk_indices[torch.arange(num_gt, device=device), random_choice]
        else:
            # Original CDN Logic: Single Nearest Neighbor
            nearest_gt_indices = torch.argmin(pairwise_dist, dim=1)

        to_replace_indices_batch = torch.nonzero(replace_mask[b]).squeeze(-1)
        for neg_idx_tiled in to_replace_indices_batch:
            orig_gt_idx = neg_idx_tiled % max_gt_num
            if orig_gt_idx >= num_gt: continue
            nearest_idx_orig = nearest_gt_indices[orig_gt_idx]
            nearest_gt_box = current_gt_boxes[nearest_idx_orig]
            input_query_bbox[b, neg_idx_tiled] = nearest_gt_box

    # --- Section 5: Noise Injection (Controlled by the new parameter) ---
    if box_noise_scale > 0:
        known_bbox_xyxy = box_cxcywh_to_xyxy(input_query_bbox)
        
        if use_tiny_noise:
            # HND Logic: Multi-level Noise
            whwh = torch.cat([input_query_bbox[..., 2:], input_query_bbox[..., 2:]], dim=-1)
            diff = whwh * 0.5
            rand_sign = (torch.randint_like(input_query_bbox, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0)
            rand_part = torch.rand_like(input_query_bbox)
            non_replaced_neg_mask = neg_mask & (~replace_mask) & pad_gt_mask
            rand_part = torch.where(non_replaced_neg_mask.unsqueeze(-1), rand_part + 1.0, rand_part)
            noise = rand_part * rand_sign * diff
            noise_scale_factor = torch.full_like(input_query_bbox, box_noise_scale)
            tiny_noise_scale = box_noise_scale * 0.1
            noise_scale_factor = torch.where(
                replace_mask.unsqueeze(-1), # Use `replace_mask` which is already defined
                torch.full_like(noise_scale_factor, tiny_noise_scale),
                noise_scale_factor
            )
            final_noise = noise * noise_scale_factor
            noise_application_mask = pad_gt_mask
            final_bbox_xyxy = torch.where(
                noise_application_mask.unsqueeze(-1),
                known_bbox_xyxy + final_noise,
                known_bbox_xyxy,
            )
        else:
            # Original CDN Logic: No noise on replaced boxes
            whwh = torch.cat([input_query_bbox[..., 2:], input_query_bbox[..., 2:]], dim=-1)
            diff = whwh * 0.5 * box_noise_scale
            rand_sign = (torch.randint_like(input_query_bbox, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0)
            rand_part = torch.rand_like(input_query_bbox)
            non_replaced_neg_mask = neg_mask & (~replace_mask) & pad_gt_mask
            rand_part = torch.where(non_replaced_neg_mask.unsqueeze(-1), rand_part + 1.0, rand_part)
            noise = rand_part * rand_sign * diff
            positive_mask = (~neg_mask) & pad_gt_mask
            noise_application_mask = positive_mask | non_replaced_neg_mask
            final_bbox_xyxy = torch.where(
                noise_application_mask.unsqueeze(-1),
                known_bbox_xyxy + noise,
                known_bbox_xyxy,
            )

        # Common post-processing for both versions
        final_bbox_xyxy = torch.clamp(final_bbox_xyxy, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(final_bbox_xyxy)
        input_query_bbox = torch.clamp(input_query_bbox, 1e-6, 1.0 - 1e-6)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)
    else:
        input_query_bbox_unact = inverse_sigmoid(torch.clamp(input_query_bbox, 1e-6, 1.0 - 1e-6))
    
    # --- Section 6, 7, 8: Logits, Attn Mask, Meta Info (This part is identical for both versions) ---
    input_query_logits = class_embed(input_query_class.to(class_embed.weight.device))
    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True
    for i in range(num_group):
        start_idx = max_gt_num * 2 * i
        end_idx = min(max_gt_num * 2 * (i + 1), num_denoising)
        if start_idx >= end_idx: continue
        next_block_start = max_gt_num * 2 * (i + 1)
        if next_block_start < num_denoising:
            attn_mask[start_idx:end_idx, next_block_start:] = True
        prev_block_end = max_gt_num * 2 * i
        if prev_block_end > 0:
            attn_mask[start_idx:end_idx, :prev_block_end] = True
            
    dn_meta = {
        "cdn_positive_idx": dn_positive_idx,
        "cdn_num_group": num_group,
        "cdn_num_split": [num_denoising, num_queries],
    }
    
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta