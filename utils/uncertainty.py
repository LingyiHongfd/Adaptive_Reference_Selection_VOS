import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
import cv2

import torch
from torch.nn import functional as NF
from torchvision.transforms import functional as TF


def calc_uncertainty(score):
    score_top, _ = score.topk(k=2, dim=1)
    uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
    uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
    _, _, h, w = uncertainty.size()
    certainty = 1 - torch.sum(uncertainty) / (h * w)
    return uncertainty, certainty


def score_mul_uncertainty(score, uncertaintity, obj_num):
    _, _, h, w = score.size()
    obj_num = obj_num[0]
    score_bg = score[:, 0, :, :].unsqueeze(0)
    score_fg = score[:, 1:obj_num + 1, :, :].reshape(1, obj_num, h, w)
    threshold = float((1 / (obj_num + 1)))
    uncertaintity_c = (uncertaintity > 0.5).float()
    score_bg_c = uncertaintity_c * score_bg * 0.2
    score_fg_c = uncertaintity_c * score_fg * 2
    uncertaintity_b = (uncertaintity <= 0.5).float()
    score_bg_b = uncertaintity_b * score_bg
    score_fg_b = uncertaintity_b * score_fg
    score_bg = score_bg_b + score_bg_c
    score_fg = score_fg_b + score_fg_c
    score = torch.cat((score_bg, score_fg), dim=1)
    return score


def select_embeddings(current_embedding, prev_embedding, prev_mask, uncertainty):
    uncertainty_t = 1 / uncertainty
    ft, c, h, w = prev_embedding[0].size()
    prev_embedding_4x, prev_embedding_8x, prev_embedding_16x = prev_embedding
    current_embedding_n = current_embedding / (torch.norm(current_embedding, dim=(2, 3), keepdim=True) + 1e-12)
    current_embedding_n = current_embedding_n.expand((ft, c, h, w))
    current_embedding_n = current_embedding_n.reshape(ft, -1, h * w)
    prev_embedding_n = prev_embedding_4x / (torch.norm(prev_embedding_4x, dim=(2, 3), keepdim=True) + 1e-12)
    prev_embedding_n = prev_embedding_n.permute(0, 2, 3, 1).reshape(-1, h * w, 256)
    featue_sim = (current_embedding_n @ prev_embedding_n)
    featue_sim = torch.mean(featue_sim.reshape(ft, -1)[:, slice(0, c * c, c + 1)], 1).unsqueeze(1)
    metric = 2 / ((1 / featue_sim) + uncertainty_t)
    leng = metric.size(0)
    quotient, remainder = divmod(leng, 4)
    index = []

    stride = 5
    block = 5

    if leng > 4:
        if leng < 13:
            slide1, slide2, slide3, slide4 = quotient + int((remainder) > 0), quotient + int(
                (remainder - 1) > 0), quotient + int((remainder - 2) > 0), quotient + int((remainder - 3) > 0)
            slide3 = slide3 + slide4
            slide2 = slide3 + slide2
            slide1 = slide1 + slide2
            slide = metric[0:slide4, ].unsqueeze(0)
            _, idx = torch.max(slide, 1)
            index.append(idx[0][0])
            slide = metric[slide4:slide3].unsqueeze(0)
            _, idx = torch.max(slide, 1)
            index.append(idx[0][0] + slide4)
            slide = metric[slide3:slide2].unsqueeze(0)
            _, idx = torch.max(slide, 1)
            index.append(idx[0][0] + slide3)
            slide = metric[slide2:slide1].unsqueeze(0)
            _, idx = torch.max(slide, 1)
            index.append(idx[0][0] + slide2)
        else:
            idx = leng + block
            slide_num = 0
            tmp_idx = []
            for i in range(4):
                if i == 1:
                    slide_num = slide_num - (block - 1) - stride + idx  # right
                if i == 0:
                    slide_num = idx - block
                if i > 1:
                    slide_num = slide_num - (block - 1) - stride + idx  # right
                if i == 0:
                    slide = metric[slide_num - block:slide_num].unsqueeze(0)
                else:
                    slide = metric[max(0, slide_num - block):max(1, slide_num)].unsqueeze(0)
                _, idx = torch.max(slide, 1)
                idx = idx[0][0]
                if i == 0:
                    tmp_idx.append(leng + idx - block)
                else:
                    if i == 1:
                        tmp_idx.append(max(0, idx + slide_num - (block - 1) - stride))
                    if i > 1:
                        tmp_idx.append(max(0, idx + slide_num - (block - 1) - stride))

            for i in range(4):
                index.append(tmp_idx[3 - i])
    else:
        if leng == 1:
            index = [0, 0, 0, 0]
        if leng == 2:
            index = [0, 0, 1, 1]
        if leng == 3:
            index = [0, 0, 1, 2]
        if leng == 4:
            index = [0, 1, 2, 3]
    prev_embeddings_4x = []
    prev_embeddings_8x = []
    prev_embeddings_16x = []
    prev_masks = []
    for i in range(4):
        idx_t = index[i]
        if i == 0:
            ref_embedding_4x = prev_embedding_4x[idx_t,].unsqueeze(0)
            ref_embedding_8x = prev_embedding_8x[idx_t,].unsqueeze(0)
            ref_embedding_16x = prev_embedding_16x[idx_t,].unsqueeze(0)
            ref_mask = prev_mask[0, idx_t].unsqueeze(0).unsqueeze(0)
        else:
            prev_embeddings_4x.append(prev_embedding_4x[idx_t,].unsqueeze(0))
            prev_embeddings_8x.append(prev_embedding_8x[idx_t,].unsqueeze(0))
            prev_embeddings_16x.append(prev_embedding_16x[idx_t,].unsqueeze(0))
            prev_masks.append(prev_mask[0, idx_t].unsqueeze(0))

    prev_embeddings_4x = torch.cat(prev_embeddings_4x, dim=1)
    prev_embeddings_8x = torch.cat(prev_embeddings_8x, dim=1)
    prev_embeddings_16x = torch.cat(prev_embeddings_16x, dim=1)
    prev_embeddings = [prev_embeddings_4x, prev_embeddings_8x, prev_embeddings_16x]
    ref_embedding = [ref_embedding_4x, ref_embedding_8x, ref_embedding_16x]
    prev_masks = torch.cat(prev_masks, dim=0).unsqueeze(0)

    del (current_embedding_n)
    del (prev_embedding_n)
    del (metric)
    del (featue_sim)
    return ref_embedding, ref_mask, prev_embeddings, prev_masks
