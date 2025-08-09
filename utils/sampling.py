import torch
from torchvision.transforms.functional import crop
import tqdm as tqdm


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond, seq, model, b, eta=0., device=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)

            et = model(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to(device))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(device))
    return xs, x0_preds

def merge_patches_with_skeleton(patches, corners, p_size, skeleton, device):
    """
    基于骨架图拼合小块，仅在此步骤使用骨架约束
    patches: 所有生成的小块列表，每个元素为 [batch, channels, p_size, p_size]
    corners: 小块坐标列表 [(hi, wi), ...]
    p_size: 小块尺寸
    skeleton: 整图骨架张量 [batch, 1, H, W]
    """
    # 获取整图尺寸（从骨架图推导）
    batch_size, _, H, W = skeleton.shape
    # 初始化输出图像和权重掩码
    merged = torch.zeros((batch_size, patches[0].shape[1], H, W), device=device)
    weight_mask = torch.zeros((batch_size, 1, H, W), device=device)  # 记录每个位置的权重总和

    # 预处理骨架图为拼合权重（骨架区域权重更高）
    skeleton = (skeleton - skeleton.min()) / (skeleton.max() - skeleton.min() + 1e-8)  # 归一化到[0,1]
    skeleton_weight = 0.5 + 1.0 * skeleton  # 骨架区域权重1.5，非骨架0.5（可调整）

    # 遍历所有小块，累加内容并记录权重
    for idx, (hi, wi) in enumerate(corners):
        patch = patches[idx]  # 当前小块
        # 提取当前小块对应位置的骨架权重
        patch_weight = crop(skeleton_weight, hi, wi, p_size, p_size)  # [batch, 1, p_size, p_size]
        
        # 累加小块内容（乘以骨架权重）
        merged[:, :, hi:hi+p_size, wi:wi+p_size] += patch * patch_weight
        # 累加权重掩码（用于后续归一化）
        weight_mask[:, :, hi:hi+p_size, wi:wi+p_size] += patch_weight

    # 用权重总和归一化，避免重叠区域过亮
    merged = merged / (weight_mask + 1e-8)  # 加小常数防止除零
    return merged



def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True,
                                  device=None, gen_diffusion=None,skeleton=None):
    with torch.no_grad():
        if gen_diffusion is not None:
            b = torch.from_numpy(gen_diffusion.betas).float().to(device)
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]


        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
        # 存储所有时间步的小块结果（最后一步用于拼合）
        final_patches = []

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)

            at = compute_alpha(b, t.long())
            if torch.cuda.is_available() and device.index is None:
                num_gpus = torch.cuda.device_count()
                copied_t = [t.clone() for _ in range(num_gpus)]
                t = torch.cat(copied_t)
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)
            et_output = torch.zeros_like(x_cond, device=x.device)

            if manual_batching:
                manual_batching_size = 64
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners],
                                         dim=0)
                for k in range(0, len(corners), manual_batching_size):
                    if model.module.learn_sigma and gen_diffusion is not None:
                        c = x.shape[1]
                        model_output = model(torch.cat([x_cond_patch[k:k + manual_batching_size],
                                                   xt_patch[k:k + manual_batching_size]], dim=1), t)
                        outputs, model_var_values = torch.split(model_output, c, dim=1)
                    else:
                        outputs = model(torch.cat([x_cond_patch[k:k + manual_batching_size],
                                                   xt_patch[k:k + manual_batching_size]], dim=1), t)
                    for idx, (hi, wi) in enumerate(corners[k:k + manual_batching_size]):
                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    x_cond_patch = data_transform(x_cond_patch)
                    if model.module.learn_sigma and gen_diffusion is not None:
                        c = x.shape[1]
                        model_output = model(torch.cat([x_cond_patch, xt_patch], dim=1), t)
                        outputs, model_var_values = torch.split(model_output, c, dim=1)
                        et_output[:, :, hi:hi + p_size, wi:wi + p_size] += outputs
                    else:
                        et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch],
                                                                                           dim=1), t)

            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to(device))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(device))
            # ---------------------- 新增代码：记录最后一步的小块 ----------------------
            if j == -1:  # 最后一个时间步，存储所有小块
                final_patches = [crop(x0_t, hi, wi, p_size, p_size) for (hi, wi) in corners]
            
        # ---------------------- 新增代码：最后阶段用骨架拼合（保留x_grid_mask） ----------------------
        if skeleton is not None and final_patches:
            # 调用拼合函数，传入x_grid_mask用于平衡重叠
            merged_output = merge_patches_with_skeleton(
                final_patches, corners, p_size, skeleton, x_grid_mask, device
            )
        else:
            # 无骨架时，用原始x_grid_mask平均拼合（保留原逻辑）
            merged_output = torch.zeros_like(x_cond, device=device)
            for idx, (hi, wi) in enumerate(corners):
                merged_output[:, :, hi:hi+p_size, wi:wi+p_size] += final_patches[idx]
            merged_output = merged_output / (x_grid_mask + 1e-8)

        # 更新最终结果
        xs[-1] = merged_output
        x0_preds[-1] = merged_output
        # ---------------------------------------------------------------------------------------


    return xs, x0_preds