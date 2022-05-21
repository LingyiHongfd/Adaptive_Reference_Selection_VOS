import os
import importlib
import time
import datetime as datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from random import randint
from dataloaders.datasets import DAVIS_Test
import dataloaders.custom_transforms as tr
from networks.deeplab.deeplab import DeepLab
from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder
from utils.uncertainty import *
from networks.contour.contour_models import contour_model
from networks.contour.contour_optim import contour_optim, bilateral_solve


class Evaluator(object):
    def __init__(self, cfg):
        self.gpu = cfg.TEST_GPU_ID
        self.cfg = cfg
        print("Use GPU {} for evaluating".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        print('Using uncertainty and feature_sim for reference select')

        self.print_log('Build backbone.')
        self.feature_extracter = DeepLab(backbone='resnet', freeze_bn=True).cuda(self.gpu)

        self.print_log('Build VOS model.')
        ADARS = importlib.import_module(cfg.MODEL_MODULE)
        self.model = ADARS.get_module()(cfg, self.feature_extracter).cuda(self.gpu)

        self.process_pretrained_model()

        self.prepare_dataset()

        self.contour_model = contour_model().cuda(self.gpu)

    def process_pretrained_model(self):
        cfg = self.cfg

        self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.gpu)

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE, cfg.TEST_FLIP, cfg.TEST_MULTISCALE),
            tr.MultiToTensor()])

        eval_name = 'davis'

        resolution = '480p'
        self.result_root = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations', resolution)
        self.dataset = DAVIS_Test(split=cfg.TEST_DATASET_SPLIT,
                                  root=cfg.DIR_DAVIS,
                                  year=2017,
                                  transform=eval_transforms,
                                  full_resolution=False, )

        self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations')
        self.zip_dir = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET, '{}.zip'.format(eval_name))
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root)
        self.print_log('Done!')

    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        video_num = 0
        total_time = 0
        total_frame = 0
        total_sfps = 0
        total_video_num = len(self.dataset)

        for seq_idx, seq_dataset in enumerate(self.dataset):
            video_num += 1
            seq_name = seq_dataset.seq_name
            print('Processing Seq {} [{}/{}]:'.format(seq_name, video_num, total_video_num))
            torch.cuda.empty_cache()

            seq_dataloader = DataLoader(seq_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

            seq_total_time = 0
            seq_total_frame = 0
            ref_embeddings = []
            ref_masks = []
            prev_embedding_4x = []
            prev_embedding_8x = []
            prev_embedding_16x = []
            prev_mask = []
            pc_mask = []
            prev_embedding_n = []
            uncertainty = torch.tensor([1]).unsqueeze(0).cuda()
            save_cache = True
            cache_length = 25
            with torch.no_grad():
                for frame_idx, input in enumerate(seq_dataloader):
                    time_start = time.time()

                    samples = input[0]
                    all_preds = []
                    join_label = None
                    for aug_idx in range(len(samples)):
                        if len(ref_embeddings) <= aug_idx:
                            ref_embeddings.append([])
                            ref_masks.append([])
                            prev_embedding_4x.append(None)
                            prev_embedding_8x.append(None)
                            prev_embedding_16x.append(None)
                            prev_mask.append(None)

                        sample = samples[aug_idx]
                        ref_emb = ref_embeddings[aug_idx]
                        ref_m = ref_masks[aug_idx]
                        prev_emb_4x = prev_embedding_4x[aug_idx]
                        prev_emb_8x = prev_embedding_8x[aug_idx]
                        prev_emb_16x = prev_embedding_16x[aug_idx]
                        prev_emb = [prev_emb_4x, prev_emb_8x, prev_emb_16x]
                        prev_m = prev_mask[aug_idx]

                        current_img = sample['current_img']
                        if 'current_label' in sample.keys():
                            current_label = sample['current_label'].cuda(self.gpu)
                        else:
                            current_label = None

                        obj_num = sample['meta']['obj_num']
                        imgname = sample['meta']['current_name']
                        ori_height = sample['meta']['height']
                        ori_width = sample['meta']['width']
                        current_img = current_img.cuda(self.gpu)
                        obj_num = obj_num.cuda(self.gpu)
                        bs, _, h, w = current_img.size()

                        all_pred, current_embedding, = self.model.forward_for_eval(
                            ref_emb, ref_m, prev_emb, prev_m, current_img, pc_mask,
                            [ori_height, ori_width], obj_num, uncertainty, )

                        contour = self.contour_model(current_img, (ori_height, ori_width))

                        if all_pred != None:
                            uncertainty_map, certainty = calc_uncertainty(all_pred)

                            mask_optimed, contour_optimed, new_certainty = contour_optim(current_img, contour[-1],
                                                                                           all_pred, uncertainty_map,
                                                                                           certainty,
                                                                                           obj_num[0])
                            all_pred = mask_optimed.unsqueeze(0)

                            if save_cache:
                                if (len(uncertainty) + 1 > cache_length):
                                    uncertainty = torch.cat(
                                        (uncertainty[1:cache_length], certainty.unsqueeze(0).unsqueeze(0)),
                                        dim=0)
                                else:
                                    uncertainty = torch.cat((uncertainty, certainty.unsqueeze(0).unsqueeze(0)), dim=0)
                            else:
                                uncertainty = torch.cat((uncertainty, certainty.unsqueeze(0).unsqueeze(0)), dim=0)

                        if frame_idx == 0:
                            if current_label is None:
                                print("No first frame label in Seq {}.".format(seq_name))
                            ref_embeddings[aug_idx].append(current_embedding)
                            ref_masks[aug_idx].append(current_label)
                            prev_embedding_4x[aug_idx] = current_embedding[0]
                            prev_embedding_8x[aug_idx] = current_embedding[1]
                            prev_embedding_16x[aug_idx] = current_embedding[2]
                            prev_mask[aug_idx] = current_label

                            pc_mask = current_label.byte()
                        else:

                            pc_mask = all_pred

                            if not sample['meta']['flip'] and not (current_label is None) and join_label is None:
                                join_label = current_label
                            all_preds.append(all_pred)
                            if current_label is not None:
                                ref_embeddings[aug_idx].append(current_embedding)
                            ref_embeddings[aug_idx][0] = prev_embedding_4x[aug_idx][0,]
                            if save_cache:
                                if len(prev_embedding_4x[aug_idx]) + 1 <= cache_length:
                                    prev_embedding_4x[aug_idx] = torch.cat(
                                        (prev_embedding_4x[aug_idx], current_embedding[0]), 0)
                                    prev_embedding_8x[aug_idx] = torch.cat(
                                        (prev_embedding_8x[aug_idx], current_embedding[1]), 0)
                                    prev_embedding_16x[aug_idx] = torch.cat(
                                        (prev_embedding_16x[aug_idx], current_embedding[2]), 0)
                                else:
                                    prev_embedding_4x[aug_idx] = torch.cat(
                                        (prev_embedding_4x[aug_idx][1:cache_length], current_embedding[0]), 0)
                                    prev_embedding_8x[aug_idx] = torch.cat(
                                        (prev_embedding_8x[aug_idx][1:cache_length], current_embedding[1]), 0)
                                    prev_embedding_16x[aug_idx] = torch.cat(
                                        (prev_embedding_16x[aug_idx][1:cache_length], current_embedding[2]), 0)
                            else:
                                prev_embedding_4x[aug_idx] = torch.cat(
                                    (prev_embedding_4x[aug_idx], current_embedding[0]), 0)
                                prev_embedding_8x[aug_idx] = torch.cat(
                                    (prev_embedding_8x[aug_idx], current_embedding[1]), 0)
                                prev_embedding_16x[aug_idx] = torch.cat(
                                    (prev_embedding_16x[aug_idx], current_embedding[2]), 0)
                            if join_label is not None:
                                ref_embeddings[aug_idx].append(current_embedding)
                                prev_embedding_4x[aug_idx] = current_embedding[0]
                                prev_embedding_8x[aug_idx] = current_embedding[1]
                                prev_embedding_16x[aug_idx] = current_embedding[2]
                                uncertainty = torch.tensor([(1 + certainty) / 2]).unsqueeze(0).cuda()
                    if frame_idx > 0:
                        all_preds = torch.cat(all_preds, dim=0)
                        all_preds = torch.mean(all_preds, dim=0)
                        pred_label = torch.argmax(all_preds, dim=0)

                        if join_label is not None:
                            join_label = join_label.squeeze(0).squeeze(0)
                            keep = (join_label == 0).long()
                            pred_label = pred_label * keep + join_label * (1 - keep)
                            pred_label = pred_label
                        current_label = pred_label.view(1, 1, ori_height, ori_width)
                        flip_pred_label = flip_tensor(pred_label, 1)
                        flip_current_label = flip_pred_label.view(1, 1, ori_height, ori_width)

                        for aug_idx in range(len(samples)):
                            if join_label is not None:
                                if samples[aug_idx]['meta']['flip']:
                                    ref_masks[aug_idx].append(flip_current_label)
                                else:
                                    ref_masks[aug_idx].append(current_label)
                            if samples[aug_idx]['meta']['flip']:
                                prev_mask[aug_idx] = flip_current_label
                            else:
                                ref_masks[aug_idx][0] = prev_mask[aug_idx][:, 0].unsqueeze(0)
                                if save_cache:
                                    if len(prev_mask[aug_idx][0]) + 1 <= cache_length:
                                        prev_mask[aug_idx] = torch.cat((prev_mask[aug_idx], current_label.byte()), 1)
                                    else:
                                        prev_mask[aug_idx] = torch.cat(
                                            (prev_mask[aug_idx][:, 1:cache_length], current_label.byte()), 1)
                                else:
                                    prev_mask[aug_idx] = torch.cat((prev_mask[aug_idx], current_label.byte()), 1)
                                if join_label is not None:
                                    pc_mask = current_label.byte()
                            if join_label is not None:
                                prev_embedding_4x[aug_idx] = current_embedding[0]
                                prev_embedding_8x[aug_idx] = current_embedding[1]
                                prev_embedding_16x[aug_idx] = current_embedding[2]
                                prev_mask[aug_idx] = current_label

                        one_frametime = time.time() - time_start
                        seq_total_time += one_frametime
                        seq_total_frame += 1
                        obj_num = obj_num[0].item()
                        print('[{}/{}] Frame: {}, Obj Num: {}, Time: {}'.format(video_num, total_video_num, imgname[0],
                                                                                obj_num, one_frametime))

                        pred_label = torch.argmax(mask_optimed, dim=0)
                        save_mask(pred_label,
                                  os.path.join(self.result_root, seq_name, imgname[0].split('.')[0] + '.png'))
                    else:
                        one_frametime = time.time() - time_start
                        seq_total_time += one_frametime
                        print('Ref Frame: {}, Time: {}'.format(imgname[0], one_frametime))
                    if visdom_vis:
                        win_pred = viz.image(current_label[aug_idx].float() * 50, win=win_pred)

                del (ref_embeddings)
                del (ref_masks)
                del (prev_embedding_4x)
                del (prev_embedding_8x)
                del (prev_embedding_16x)
                del (prev_mask)
                del (pc_mask)
                del (uncertainty)
                del (prev_embedding_n)
                del (seq_dataset)
                del (seq_dataloader)
                del (all_preds)

            seq_avg_time_per_frame = seq_total_time / seq_total_frame
            total_time += seq_total_time
            total_frame += seq_total_frame
            total_avg_time_per_frame = total_time / total_frame
            total_sfps += seq_avg_time_per_frame
            avg_sfps = total_sfps / (seq_idx + 1)
            print("Seq {} FPS: {}, Total FPS: {}, FPS per Seq: {}".format(seq_name, 1. / seq_avg_time_per_frame,
                                                                          1. / total_avg_time_per_frame, 1. / avg_sfps))

        zip_folder(self.source_folder, self.zip_dir)
        self.print_log('Save result to {}.'.format(self.zip_dir))

    def print_log(self, string):
        print(string)
