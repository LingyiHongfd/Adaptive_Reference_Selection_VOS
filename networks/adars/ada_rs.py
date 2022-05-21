import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.matching import global_matching, global_matching_for_eval, foreground2background
from networks.layers.attention import calculate_attention_head_for_eval
from networks.adars.ensembler import DynamicPreHead, CollaborativeEnsemblerMS
from utils.uncertainty import *
from networks.layers.fpn import FPN
from networks.contour.contour_models import contour_model, Contour_Conv
from networks.layers.self_attention import *


class ADARS(nn.Module):
    def __init__(self, cfg, feature_extracter):
        super(ADARS, self).__init__()
        self.cfg = cfg
        self.epsilon = cfg.MODEL_EPSILON

        self.feature_extracter = feature_extracter

        self.contour_model = contour_model().cuda()

        self.fpn = FPN(in_dim_4x=256, in_dim_8x=512, in_dim_16x=256, out_dim=256)

        self.contour_conv = Contour_Conv()
        self.seperate_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, kernel_size=3, stride=1, padding=1,
                                       groups=cfg.MODEL_ASPP_OUTDIM)
        self.bn1 = nn.GroupNorm(cfg.MODEL_GN_GROUPS, cfg.MODEL_ASPP_OUTDIM)
        self.relu1 = nn.ReLU(True)
        self.embedding_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_SEMANTIC_EMBEDDING_DIM, 1, 1)
        self.bn2 = nn.GroupNorm(cfg.MODEL_GN_EMB_GROUPS, cfg.MODEL_SEMANTIC_EMBEDDING_DIM)
        self.relu2 = nn.ReLU(True)
        self.semantic_embedding = nn.Sequential(
            *[self.seperate_conv, self.bn1, self.relu1, self.embedding_conv, self.bn2, self.relu2])

        self.bg_bias = nn.Parameter(torch.zeros(3, 1, 1, 1))
        self.fg_bias = nn.Parameter(torch.zeros(3, 1, 1, 1))

        for m in self.semantic_embedding:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.dynamic_seghead = CollaborativeEnsemblerMS(
            in_dim_4x=832,
            in_dim_8x=832,
            in_dim_16x=832,
            attention_dim=1112,
            embed_dim=cfg.MODEL_HEAD_EMBEDDING_DIM,
            refine_dim=cfg.MODEL_REFINE_CHANNELS,
            low_level_dim=cfg.MODEL_LOW_LEVEL_INPLANES)

        in_dim = 2 + len(cfg.MODEL_MULTI_LOCAL_DISTANCE)
        if cfg.MODEL_MATCHING_BACKGROUND:
            in_dim += 1 + len(cfg.MODEL_MULTI_LOCAL_DISTANCE)
        self.dynamic_prehead_4x = DynamicPreHead(
            in_dim=11,
            embed_dim=cfg.MODEL_PRE_HEAD_EMBEDDING_DIM)
        self.dynamic_prehead_8x = DynamicPreHead(
            in_dim=2,
            embed_dim=cfg.MODEL_PRE_HEAD_EMBEDDING_DIM)
        self.dynamic_prehead_16x = DynamicPreHead(
            in_dim=10,
            embed_dim=cfg.MODEL_PRE_HEAD_EMBEDDING_DIM)

        self.self_attention_4xf = Self_Attention(256)
        self.self_attention_4xb = Self_Attention(256)
        self.self_attention_8xf = Self_Attention(256)
        self.self_attention_8xb = Self_Attention(256)
        self.self_attention_16xf = Self_Attention(256)
        self.self_attention_16xb = Self_Attention(256)

    def forward_for_eval(self, ref_embeddings, ref_masks, prev_embedding, prev_mask, current_frame, pc_mask, pred_size,
                         gt_ids, uncertainty, ):
        current_frame_embedding_4x, current_frame_embedding_8x, current_frame_embedding_16x, current_low_level = self.extract_feature(
            current_frame)
        current_frame_embedding = [current_frame_embedding_4x, current_frame_embedding_8x, current_frame_embedding_16x]

        if prev_embedding[0] is None:
            return None, current_frame_embedding
        else:
            contour = self.contour_model(current_frame, pred_size)[-1]
            ref_embedding, ref_masks, prev_embeddings, prev_mask, joint, joint_sim, delta, status = select_embeddings(
                current_frame_embedding[0], prev_embedding, prev_mask, uncertainty)
            bs, c, h, w = current_frame_embedding[0].size()
            tmp_dic, _ = self.before_seghead_process(
                ref_embedding,
                prev_embeddings,
                current_frame_embedding,
                ref_masks,
                prev_mask,
                pc_mask,
                contour,
                gt_ids,
                current_low_levels=current_low_level, )
            all_pred = []
            for i in range(bs):
                pred = tmp_dic[i]
                pred = nn.functional.interpolate(pred, size=(pred_size[0], pred_size[1]), mode='bilinear',
                                                 align_corners=True)
                all_pred.append(pred)
            all_pred = torch.cat(all_pred, dim=0)

            all_pred = torch.softmax(all_pred, dim=1)
            return all_pred, current_frame_embedding

    def extract_feature(self, x):
        x, aspp_x, low_level, mid_level = self.feature_extracter(x)

        x_4x, x_8x, x_16x = self.fpn(x, mid_level, aspp_x)

        return x_4x, x_8x, x_16x, low_level

    def before_seghead_process(self,
                               ref_frame_embeddings=None, previous_frame_embeddings=None,
                               current_frame_embeddings=None,
                               ref_frame_label=None, previous_frame_mask=None, pc_frame_mask=None, contour=None,
                               gt_ids=None, current_low_levels=None, ):
        cfg = self.cfg
        dic_tmp = []
        bs, c, h, w = current_frame_embeddings[0].size()

        scale_ref_frame_labels = []
        scale_previous_frame_labels = []
        for current_frame_embedding in current_frame_embeddings:
            bs, c, h, w = current_frame_embedding.size()

            scale_ref_frame_label = torch.nn.functional.interpolate(ref_frame_label.float(),
                                                                    size=(h, w), mode='nearest')
            scale_ref_frame_label = scale_ref_frame_label.int()
            scale_ref_frame_labels.append(scale_ref_frame_label)

            scale_previous_frame_label = torch.nn.functional.interpolate(previous_frame_mask.float(), size=(h, w),
                                                                         mode='nearest')
            scale_previous_frame_labels.append(scale_previous_frame_label.int())

        boards = {'image': {}, 'scalar': {}}

        to_cats = []
        attentions = []

        for n in range(bs):
            ref_obj_ids = torch.arange(0, gt_ids[n] + 1, device=current_frame_embedding.device).int().view(-1, 1, 1, 1)
            obj_num = ref_obj_ids.size(0)

            for idx in range(3):
                bs, c, h, w = current_frame_embeddings[idx].size()

                if idx == 1:

                    to_cat_current_frame_embedding = current_frame_embeddings[idx].expand(
                        (obj_num, -1, -1, -1))
                    to_cat_prev_frame_embedding = previous_frame_embeddings[idx][n, 256 * 2:256 * 3].unsqueeze(
                        0).expand((obj_num, -1, -1, -1))

                    seq_previous_frame_label = (scale_previous_frame_labels[idx][n, -1].int() == ref_obj_ids).float()
                    seq_ref_frame_label = (scale_ref_frame_labels[idx].int() == ref_obj_ids).float()
                    to_cat_ref_frame = seq_ref_frame_label

                    if pc_frame_mask != None:
                        if pc_frame_mask.size(1) == 1:
                            pre_to_cat_label = torch.nn.functional.interpolate(pc_frame_mask.float(), size=(h, w),
                                                                               mode='nearest')
                            pre_to_cat_label = (pre_to_cat_label.int() == ref_obj_ids).float()
                        else:
                            pre_to_cat_label = torch.nn.functional.interpolate(pc_frame_mask.float(), size=(h, w),
                                                                               mode='nearest')
                            pre_to_cat_label = pre_to_cat_label.permute(1, 0, 2, 3)

                    pre_to_cat = torch.cat((seq_previous_frame_label, pre_to_cat_label), 1)

                    pre_to_cat = self.dynamic_prehead_8x(pre_to_cat)

                    to_cat_prev_frame_embedding_fg = self.self_attention_16xf(to_cat_prev_frame_embedding,
                                                                              seq_previous_frame_label)
                    to_cat_prev_frame_embedding_bg = self.self_attention_16xb(to_cat_prev_frame_embedding, (
                            1 - seq_previous_frame_label))

                    to_cat = torch.cat((to_cat_current_frame_embedding, to_cat_prev_frame_embedding_fg,
                                        to_cat_prev_frame_embedding_bg, pre_to_cat), 1)
                    to_cats.append(to_cat)

                    seq_ref_frame_labels = []
                    scale_ref_frame_label = scale_ref_frame_labels[idx]

                    seq_ref_frame_label = (scale_ref_frame_label.int() == ref_obj_ids).float()
                    seq_ref_frame_labels.append(seq_ref_frame_label)

                    to_cat_previous_frame = []
                    for u in range(3):
                        seq_previous_frame_label = (scale_previous_frame_labels[idx][n, u].int() == ref_obj_ids).float()
                        to_cat_previous_frame_b = seq_previous_frame_label
                        to_cat_previous_frame.append(to_cat_previous_frame_b)
                    to_cat_previous_frame = torch.cat(to_cat_previous_frame, 1)

                    attention_head = calculate_attention_head_for_eval(
                        [ref_frame_embeddings[idx]],
                        [seq_ref_frame_labels[0]],
                        previous_frame_embeddings[idx].expand((obj_num, -1, -1, -1)),
                        to_cat_previous_frame,
                        epsilon=self.epsilon)

                    attentions.append(attention_head)


                else:
                    if gt_ids[n] > 0:
                        dis_bias = torch.cat([self.bg_bias[idx].unsqueeze(0),
                                              self.fg_bias[idx].unsqueeze(0).expand(gt_ids[n], -1, -1, -1)],
                                             dim=0)
                    else:
                        dis_bias = self.bg_bias[idx].unsqueeze(0)

                    seq_current_frame_embedding = current_frame_embeddings[idx].squeeze(0)
                    seq_current_frame_embedding = seq_current_frame_embedding.permute(1, 2, 0)
                    ########################Global dist map

                    all_reference_embeddings = []
                    all_reference_labels = []
                    seq_ref_frame_labels = []

                    each_ref_frame_embedding = ref_frame_embeddings[idx]
                    scale_ref_frame_label = scale_ref_frame_labels[idx]

                    seq_ref_frame_embedding = each_ref_frame_embedding.squeeze(0)
                    seq_ref_frame_embedding = seq_ref_frame_embedding.permute(1, 2, 0)
                    all_reference_embeddings.append(seq_ref_frame_embedding)

                    seq_ref_frame_label = (scale_ref_frame_label.int() == ref_obj_ids).float()
                    seq_ref_frame_labels.append(seq_ref_frame_label)
                    seq_ref_frame_label = seq_ref_frame_label.squeeze(1).permute(1, 2, 0)
                    all_reference_labels.append(seq_ref_frame_label)
                    if idx == 0:
                        global_matching_fg = global_matching_for_eval(
                            all_reference_embeddings=all_reference_embeddings,
                            query_embeddings=seq_current_frame_embedding,
                            all_reference_labels=all_reference_labels,
                            n_chunks=cfg.TEST_GLOBAL_CHUNKS,
                            dis_bias=dis_bias,
                            atrous_rate=2,
                            use_float16=cfg.MODEL_FLOAT16_MATCHING)
                    if idx == 2:
                        global_matching_fg = global_matching_for_eval(
                            all_reference_embeddings=all_reference_embeddings,
                            query_embeddings=seq_current_frame_embedding,
                            all_reference_labels=all_reference_labels,
                            n_chunks=cfg.TEST_GLOBAL_CHUNKS,
                            dis_bias=dis_bias,
                            atrous_rate=cfg.TEST_GLOBAL_ATROUS_RATE,
                            use_float16=cfg.MODEL_FLOAT16_MATCHING)

                    #########################Local dist map
                    local_matching_fg = []
                    to_cat_previous_frame = []
                    for u in range(int(previous_frame_embeddings[0].size()[1] / 256)):
                        seq_prev_frame_embedding = previous_frame_embeddings[idx][n, u * 256:(u + 1) * 256]
                        seq_prev_frame_embedding = seq_prev_frame_embedding.permute(1, 2, 0)
                        seq_previous_frame_label = (scale_previous_frame_labels[idx][n, u].int() == ref_obj_ids).float()
                        to_cat_previous_frame_b = seq_previous_frame_label
                        seq_previous_frame_label = seq_previous_frame_label.squeeze(1).permute(1, 2, 0)
                        if idx == 0:
                            local_matching_fg_b = global_matching(
                                reference_embeddings=seq_prev_frame_embedding,
                                query_embeddings=seq_current_frame_embedding,
                                reference_labels=seq_previous_frame_label,
                                n_chunks=cfg.TRAIN_GLOBAL_CHUNKS,
                                dis_bias=dis_bias,
                                atrous_rate=2,
                                use_float16=cfg.MODEL_FLOAT16_MATCHING)

                        if idx == 2:
                            local_matching_fg_b = global_matching(
                                reference_embeddings=seq_prev_frame_embedding,
                                query_embeddings=seq_current_frame_embedding,
                                reference_labels=seq_previous_frame_label,
                                n_chunks=cfg.TRAIN_GLOBAL_CHUNKS,
                                dis_bias=dis_bias,
                                atrous_rate=cfg.TRAIN_GLOBAL_ATROUS_RATE,
                                use_float16=cfg.MODEL_FLOAT16_MATCHING)

                        local_matching_fg.append(local_matching_fg_b)
                        to_cat_previous_frame.append(to_cat_previous_frame_b)
                        # print ('local match',local_matching_fg_b.size())
                    ##########################
                    local_matching_fg = torch.cat(local_matching_fg, -1)
                    to_cat_previous_frame = torch.cat(to_cat_previous_frame, 1)

                    to_cat_current_frame_embedding = current_frame_embeddings[idx].expand(
                        (obj_num, -1, -1, -1))
                    to_cat_prev_frame_embedding = previous_frame_embeddings[idx][n, 256 * 2:256 * 3].unsqueeze(
                        0).expand((obj_num, -1, -1, -1))

                    to_cat_global_matching_fg = global_matching_fg.squeeze(0).permute(2, 3, 0, 1)
                    to_cat_local_matching_fg = local_matching_fg.squeeze(0).permute(2, 3, 0, 1)

                    if cfg.MODEL_MATCHING_BACKGROUND:
                        to_cat_global_matching_bg = foreground2background(to_cat_global_matching_fg, gt_ids[n] + 1)
                        reshaped_prev_nn_feature_n = to_cat_local_matching_fg.permute(0, 2, 3, 1).unsqueeze(1)
                        to_cat_local_matching_bg = foreground2background(reshaped_prev_nn_feature_n, gt_ids[n] + 1)
                        to_cat_local_matching_bg = to_cat_local_matching_bg.permute(0, 4, 2, 3, 1).squeeze(-1)

                    if pc_frame_mask != None:
                        if pc_frame_mask.size(1) == 1:
                            pre_to_cat_label = torch.nn.functional.interpolate(pc_frame_mask.float(), size=(h, w),
                                                                               mode='nearest')
                            pre_to_cat_label = (pre_to_cat_label.int() == ref_obj_ids).float()
                        else:
                            pre_to_cat_label = torch.nn.functional.interpolate(pc_frame_mask.float(), size=(h, w),
                                                                               mode='nearest')
                            pre_to_cat_label = pre_to_cat_label.permute(1, 0, 2, 3)

                    to_cat_previous_frame_ = to_cat_previous_frame[:, 2, :, :].unsqueeze(1)
                    pre_to_cat = torch.cat(
                        (to_cat_global_matching_fg, to_cat_local_matching_fg, to_cat_previous_frame_, pre_to_cat_label),
                        1)
                    if cfg.MODEL_MATCHING_BACKGROUND:
                        pre_to_cat = torch.cat([pre_to_cat, to_cat_local_matching_bg, to_cat_global_matching_bg], 1)
                    if idx == 0:
                        contour = nn.functional.interpolate(contour, size=(h, w), mode='bilinear', align_corners=False)
                        contour = contour.expand(obj_num, -1, -1, -1)
                        pre_to_cat = torch.cat([pre_to_cat, contour], dim=1)
                        pre_to_cat = self.dynamic_prehead_4x(pre_to_cat)
                    if idx == 2:
                        pre_to_cat = self.dynamic_prehead_16x(pre_to_cat)
                    to_cat_prev_frame_embedding_fg = []
                    to_cat_prev_frame_embedding_bg = []
                    if idx == 0:
                        to_cat_prev_frame_embedding_fg_b = self.self_attention_4xf(to_cat_prev_frame_embedding,
                                                                                   to_cat_previous_frame_)
                        to_cat_prev_frame_embedding_bg_b = self.self_attention_4xb(to_cat_prev_frame_embedding, (
                                1 - to_cat_previous_frame_))
                        to_cat_prev_frame_embedding_fg.append(to_cat_prev_frame_embedding_fg_b)
                        to_cat_prev_frame_embedding_bg.append(to_cat_prev_frame_embedding_bg_b)
                    if idx == 2:
                        to_cat_prev_frame_embedding_fg_b = self.self_attention_16xf(to_cat_prev_frame_embedding,
                                                                                    to_cat_previous_frame_)
                        to_cat_prev_frame_embedding_bg_b = self.self_attention_16xb(to_cat_prev_frame_embedding, (
                                1 - to_cat_previous_frame_))
                        to_cat_prev_frame_embedding_fg.append(to_cat_prev_frame_embedding_fg_b)
                        to_cat_prev_frame_embedding_bg.append(to_cat_prev_frame_embedding_bg_b)
                    to_cat_prev_frame_embedding_fg = torch.cat(to_cat_prev_frame_embedding_fg, 1)
                    to_cat_prev_frame_embedding_bg = torch.cat(to_cat_prev_frame_embedding_bg, 1)

                    to_cat = torch.cat((to_cat_current_frame_embedding, to_cat_prev_frame_embedding_fg,
                                        to_cat_prev_frame_embedding_bg, pre_to_cat), 1)
                    to_cats.append(to_cat)

                    attention_head = calculate_attention_head_for_eval(
                        [ref_frame_embeddings[idx]],
                        [seq_ref_frame_labels[0]],
                        previous_frame_embeddings[idx].expand((obj_num, -1, -1, -1)),
                        to_cat_previous_frame,
                        epsilon=self.epsilon)

                    attentions.append(attention_head)

            current_low_levels = current_low_levels.expand((obj_num, -1, -1, -1))

            pred = self.dynamic_seghead(to_cats, attentions, current_low_levels)

            dic_tmp.append(pred)

        return dic_tmp, boards


def get_module():
    return ADARS
