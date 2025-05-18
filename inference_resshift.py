#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import argparse
from pathlib import Path
from omegaconf import OmegaConf
from sampler import ResShiftSampler


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--seed", type=int, default=666, help="Random seed.")
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256, 64],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--chop_stride",
            type=int,
            default=-1,
            help="Chopping stride.",
            )
    args = parser.parse_args()

    return args

def get_configs(args):

    # nacti conf
    configs = OmegaConf.load('configs/swinunet_realesrgan256.yaml')

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64) * (4 // args.scale)
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * (4 // args.scale)
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * (4 // args.scale)
        else:
            raise ValueError("Chop size must be in [512, 256]")
    else:
        chop_stride = args.chop_stride * (4 // args.scale)
    args.chop_size *= (4 // args.scale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    return configs, chop_stride

def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)
    resshift_sampler = ResShiftSampler(
            configs,
            sf=args.scale,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get('lq_size', 64),
            )
    resshift_sampler.inference(
            args.in_path,
            args.out_path,
            mask_path=None,
            bs=1,  # batch size
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
