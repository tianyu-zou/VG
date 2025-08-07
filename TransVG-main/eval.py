import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from PIL import ImageDraw
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=0., type=float)
    parser.add_argument('--lr_visu_cnn', default=0., type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)
    
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='TransVG',
                        help="Name of model to be exploited.")
    
    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--use_vl_type_embed', action='store_true',
                        help="If true, use vl_type embedding")
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evalutaion options
    parser.add_argument('--eval_set', default='test', type=str)
    parser.add_argument('--eval_model', default='/hy-tmp/zty/VG/TransVG-main/outputs/TransVG_R101_unc.pth', type=str)

    return parser

def xywh2xyxy(box):
    # [cx, cy, w, h] -> [x1, y1, x2, y2]
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy2xywh(box):
    # [x1, y1, x2, y2] -> [cx, cy, w, h]
    x1, y1, x2, y2 = box.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)

def recover_box_from_padded_image(normalized_box, padded_size, original_size):
    """
    Args:
        normalized_box: Tensor [N, 4], format [cx, cy, w, h], values in [0, 1]
        padded_size: Tuple (H_pad, W_pad), e.g., (640, 640)
        original_size: Tuple (H_orig, W_orig), original image size before any resizing or padding

    Returns:
        box in absolute coords on original image, format [x1, y1, x2, y2]
    """
    H_pad, W_pad = padded_size
    H_orig, W_orig = original_size

    # Step 1: 从归一化 box 恢复到 padded 图像坐标
    box_abs = normalized_box.clone()
    box_abs[:, 0] *= W_pad
    box_abs[:, 1] *= H_pad
    box_abs[:, 2] *= W_pad
    box_abs[:, 3] *= H_pad
    box_xyxy = xywh2xyxy(box_abs)

    # Step 2: 计算 resize scale 和 padding
    scale = min(W_pad / W_orig, H_pad / H_orig)
    new_W, new_H = int(W_orig * scale), int(H_orig * scale)

    pad_w = W_pad - new_W
    pad_h = H_pad - new_H
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    # Step 3: 去掉 padding，恢复到 resize 后尺寸
    box_xyxy[:, [0, 2]] -= pad_left
    box_xyxy[:, [1, 3]] -= pad_top

    # Step 4: 除以 scale，恢复到原图尺寸
    box_xyxy = box_xyxy / scale

    return box_xyxy.clamp(min=0, max=max(H_orig, W_orig))  # 可选裁剪



@torch.no_grad()
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)
    
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    # output log
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_log.txt").open("a") as f:
            f.write(str(args) + "\n")
    
    start_time = time.time()
    model.eval()

    pred_box_list = []
    results = []
    for _, batch in enumerate(tqdm(data_loader_test)):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        output = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        # 保存图片
        x, y, w, h = output[0]
        new_x, new_y, new_w, new_h = (int(640 * x - 0.5 * 640 * w), int(640 * y - 0.5 * 640 * h), int(640 * w), int(640 * h))
        # print(data_loader_test.dataset.images[_][2])
        # print(data_loader_test.dataset.images[_][0])
        # print(new_x, new_y, new_w, new_h)
        result = {
            "image_path": data_loader_test.dataset.images[_][0].replace('/', '\\'),  # 按照你提供的输出保持反斜杠
            "question": data_loader_test.dataset.images[_][2],
            "result": [
                [float(new_x), float(new_y)],
                [float(new_w), float(new_h)]
            ]
        }
        print(result)
        results.append(result)
        # 保存到 JSON 文件
    with open('TransVG_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
