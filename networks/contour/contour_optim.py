from networks.contour.union_find_set import *
import torch
from networks.contour.union_find_set import segments
import cv2
from skimage.io import imread

from networks.layers.bilateral_solver import BilateralSolver, BilateralGrid


def Denorm_Img_Show(sample):
    sample = sample.permute(0, 2, 3, 1)
    sample = sample * torch.Tensor([0.229, 0.224, 0.225])
    sample = sample + torch.Tensor([0.485, 0.456, 0.406])
    sample = sample * 255.
    sample = sample.permute(0, 3, 1, 2).int().float()[:, [1, 2, 0], :, :]

    return sample


def contour_group():
    edge_pred = uncertainty.squeeze(0).squeeze(0).cpu().detach().numpy()
    segment, groups = segments(edge_pred)

    return segment, groups


def bilateral_solve(img, mask, obj_nums):
    out_mask = torch.zeros_like(mask)
    img = torch.nn.functional.interpolate(img, size=(mask.size(-1), mask.size(-2)), mode='bilinear')
    img = Denorm_Img_Show(img.cpu()).cuda().permute(0, 3, 2, 1).squeeze(0).cpu().numpy()
    grid_params = {
        'sigma_luma': 4,  # Brightness bandwidth
        'sigma_chroma': 4,  # Color bandwidth
        'sigma_spatial': 8  # Spatial bandwidth
    }
    bs_params = {
        'lam': 128,  # The strength of the smoothness parameter
        'A_diag_min': 1e-5,  # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
        'cg_tol': 1e-5,  # The tolerance on the convergence in PCG
        'cg_maxiter': 25  # The number of PCG iterations
    }
    grid = BilateralGrid(img, **grid_params)
    im_shape = img.shape[:2]

    confidence = imread('/home/guopx/hly/tip/full_CFBI_c1/utils/confidence.png')
    c = np.resize(confidence, im_shape).reshape(-1, 1).astype(np.double)
    for i in range(obj_nums + 1):
        t = mask[:, i, :, :].cpu().numpy().reshape(-1, 1).astype(np.double)
        tc_filt = grid.filter(t * c)
        c_filt = grid.filter(c)
        output_filter = (tc_filt / c_filt).reshape(im_shape)
        output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)
        out_mask[:, i:, :] = torch.from_numpy(output_filter)
    torch.softmax(out_mask, dim=1)

    return out_mask


def contour_optim(current_img, contour, all_pred, uncertainty_map, certrainty, obj_num):
    hw_skip = 10
    ori_height, ori_weight = ori_size
    height_skip, width_skip = int(ori_height / hw_skip), int(ori_width / hw_skip)

    heavy_list = []
    for hi in range(hw_skip):
        for wi in range(hw_skip):
            value_msk = torch.zeros((ori_height, ori_width)).cuda()
            value_msk[int(hi * height_skip):int((hi + 1) * height_skip),
            int(wi * width_skip):int((wi + 1) * width_skip)] = 1.0
            value_msk = value_msk.float()
            uncertainty_map_uc = (uncertainty_map * value_msk > 0.7)
            leng = len(torch.nonzero(uncertainty_map_uc))
            if (leng / (height_skip * width_skip) > 0.05):
                heavy_list.append([hi, wi])

    all_pred_ = bilateral_solve(current_img, all_pred, obj_num[0])

    uncertainty_map[:, :, 0:ori_height:height_skip, 0:ori_width:width_skip] = 0
    zeros_heavy_map = torch.zeros((ori_height, ori_width)).cuda()
    if (len(heavy_list) >= 1):
        for hhi in range(len(heavy_list)):
            hi, wi = heavy_list[hhi]
            zeros_heavy_map[hi * height_skip:(hi + 1) * height_skip:1,
            wi * width_skip:(wi + 1) * width_skip:1] = 1.0
    zeros_heavy_map = zeros_heavy_map.unsqueeze(0).unsqueeze(0)

    uncertainty_map_uc = uncertainty_map * zeros_heavy_map

    mask_optimed, contour_optimed, new_certrainty = _contour_optim(current_img, contour[-1],
                                                                    all_pred, uncertainty_map_uc,
                                                                    all_pred_,
                                                                    certrainty,
                                                                    obj_num[0])
    return mask_optimed, contour_optimed, new_certrainty


def _contour_optim(img, contour, mask, uncertainty_map_hv, fbs_score, certrainty, obj_num):
    edge_pred = contour.squeeze(0).squeeze(0).cpu().detach().numpy()
    segment, groups = segments(edge_pred)

    edge = np.zeros(edge_pred.shape, np.uint8)
    for i in range(1, edge.shape[0] - 1):
        for j in range(1, edge.shape[1] - 1):
            dr = [-1, 1, 0, 0]
            dc = [0, 0, -1, 1]
            for p in range(4):
                i_, j_ = dr[p] + i, dc[p] + j
                if segment[i, j] != segment[i_, j_]:
                    for x in range(0, 1):
                        for y in range(0, 1):
                            edge[i + x, j + y] = 1

    segment = torch.from_numpy(segment)
    mask = torch.mean(mask, dim=0)

    mask_optim, new_certrainty = group_optim_smp(segment, groups, mask, uncertainty_map_hv, fbs_score, certrainty,
                                                  obj_num)
    return mask_optim, torch.from_numpy(edge) * 255, new_certrainty


def group_optim_smp(segment, groups, mask, uncertainty_map_hv, fbs_score, certrainty, obj_num):
    uncertainty_map_hv = uncertainty_map_hv.squeeze(0).squeeze(0)
    fbs_score = fbs_score.squeeze(0)
    mask = mask.clone()
    arg_mask = torch.argmax(mask, dim=0)
    _, h, w = mask.size()
    groups_len = len(groups)
    new_certrainty = 0
    for i in range(groups_len):
        group_value = groups[i]
        tmp_m = (segment == group_value).byte()
        tmp_ms = tmp_m.unsqueeze(0).repeat(obj_num + 1, 1, 1)
        position = torch.nonzero(tmp_m, as_tuple=False)
        group_nums = len(position)
        tmp_uc = uncertainty_map_hv[tmp_m]
        uc_nums = torch.nonzero(tmp_uc, as_tuple=False)
        tmp_amask = arg_mask[tmp_m]
        msk_nums = torch.nonzero(tmp_amask, as_tuple=False)
        if (len(msk_nums) == 0):
            continue
        if len(uc_nums) <= 1000:
            if (len(torch.unique(msk_nums)) <= 2):
                continue
            else:
                mask[tmp_ms] = (mask[tmp_ms] + fbs_score[tmp_ms]) / 2
                continue
        else:
            group_nums_obj = []
            sum_obj_num = 0
            for b in range(obj_num + 1):
                tmp_mask_b = tmp_amask[tmp_amask == (b)]
                tmp_num_obj = len(torch.nonzero(tmp_mask_b))
                sum_obj_num = sum_obj_num + tmp_num_obj
                group_nums_obj.append(torch.tensor(tmp_num_obj).unsqueeze(0))
            bg_nums = group_nums - sum_obj_num
            group_nums_obj[0] = group_nums_obj[0] + bg_nums
            bg_pct = bg_nums / group_nums
            if (bg_pct > 0.8):
                mask[tmp_ms] = (mask[tmp_ms] + fbs_score[tmp_ms]) / 2
                continue
            else:
                if (len(torch.unique(tmp_amask)) <= 2):
                    continue
                else:
                    _, indices = torch.sort(torch.cat(group_nums_obj, 0), 0, descending=True)
                    fg_max_idx = 0
                    fg_second_idx = 0

                    for ii in range(len(indices)):
                        if indices[ii] != 0 and group_nums_obj[indices[ii]] != 0:
                            if fg_max_idx == 0:
                                fg_max_idx = indices[ii]
                            else:
                                fg_second_idx = indices[ii]
                                break
                    if indices[0] == 0:
                        fg_max_idx = 0
                    if fg_max_idx == 0 or fg_second_idx == 0:
                        continue
                    else:
                        fg_max_nums = group_nums_obj[fg_max_idx]
                        fg_second_nums = group_nums_obj[fg_second_idx]
                        if ((fg_max_nums / group_nums) > 0.5):
                            tmp_uc = torch.zeros_like(mask).cuda()
                            print('idx', fg_max_idx, fg_second_idx)
                            for ii in range(obj_num + 1):
                                if ii == fg_second_idx or ii == fg_max_idx:
                                    if ii == fg_max_idx:
                                        tmp_uc[fg_max_idx, :, :] = mask[fg_max_idx, :, :] + mask[fg_second_idx, :, :]
                                    else:
                                        tmp_uc[fg_second_idx, :, :] = mask[fg_second_idx, :, :] / 2
                                else:
                                    tmp_uc[ii, :, :] = mask[ii, :, :]
                            mask[tmp_ms] = tmp_uc[tmp_ms]

                        continue

        mask = torch.softmax(mask, dim=1)

    return mask, new_certrainty


def group_optim(segment, groups, mask, certrainty, obj_num, f_num):
    h, w = mask.size()
    groups_len = len(groups)
    new_mask = mask.clone()
    new_certrainty = 0
    for i in range(groups_len):
        group_value = groups[i]
        tmp_m = (segment == group_value).byte()
        position = torch.nonzero(tmp_m, as_tuple=False)
        group_nums = len(position)
        tmp_mask = mask[tmp_m]
        group_nums_obj = []
        sum_obj_num = 0
        for b in range(obj_num):
            tmp_mask_b = tmp_mask[tmp_mask == (b + 1)]
            tmp_num_obj = len(torch.nonzero(tmp_mask_b))
            sum_obj_num = sum_obj_num + tmp_num_obj
            group_nums_obj.append(torch.tensor(tmp_num_obj).unsqueeze(0))
        _, fg_max_idx = torch.max(torch.cat(group_nums_obj, 0), 0)
        fg_max_nums = group_nums_obj[fg_max_idx]
        bg_nums = group_nums - sum_obj_num
        bg_pct = bg_nums / group_nums
        new_certrainty_b = max(bg_nums, fg_max_nums) / (h * w)
        if fg_max_nums / group_nums < 0.8 and fg_max_nums / group_nums > 0.6:
            # if fg_max_nums/group_nums>0.8:
            new_certrainty_b = group_nums / (h * w * 2)
            new_mask[tmp_m] = fg_max_idx.item() + 1
        if bg_pct > 0.6:
            new_mask[tmp_m] = 0
        new_certrainty = new_certrainty + new_certrainty_b
    return new_mask, new_certrainty
