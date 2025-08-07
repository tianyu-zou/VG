import argparse
import datetime
import json
import random
import time
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # batch_size max=768，0、1在用
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
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
    parser.add_argument('--lr_visual', default=0., type=float)
    # parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
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
    parser.add_argument('--is_eliminate', action='store_true')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='EEVG',
                        help="Name of model to be exploited.")

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--backbone', default='ViTDet', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=448, type=int, help='image size')
    parser.add_argument('--use_vl_type_embed', action='store_true',
                        help="If true, use vl_type embedding")
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=768, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=3, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--eliminated_threshold', default=0.015, type=float)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='../VG',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='mask_data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='gref_umd', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=40, type=int,
                        help='maximum time steps (lang length) per batch')

    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--is_segment', action='store_true', help='if use segmentation')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evalutaion options
    parser.add_argument('--eval_set', default='test', type=str)
    parser.add_argument('--eval_model', default='/hy-tmp/zty/VG/EEVG-main/outputs/mixed_coco_decoder_ViTDet_best_mask_checkpoint.pth', type=str)

    return parser


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

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_test = torch.utils.data.BatchSampler(
        sampler_test, args.batch_size, drop_last=False)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    # output log
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_log.txt").open("a") as f:
            f.write(str(args) + "\n")

    start_time = time.time()

    pred_box_list = []
    results = []
    for _, batch in enumerate(tqdm(data_loader_test)):
        img_data, text_data, target = batch

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        with torch.no_grad():
            output = model(img_data, text_data)

        x, y, w, h = output[0]
        new_x, new_y, new_w, new_h = (
        int(448 * x - 0.5 * 448 * w), int(448 * y - 0.5 * 448 * h), int(448 * w), int(448 * h))
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
    with open('EEVG_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EEVG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
