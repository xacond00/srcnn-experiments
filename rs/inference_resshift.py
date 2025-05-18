#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import math, random
from os import path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

import numpy as np
from pathlib import Path
from contextlib import nullcontext

from utils import util_net
from utils import util_image
from utils import util_common
from utils.util_image import ImageSpliterTh, batch_SSIM, batch_PSNR

from datapipe.base import BaseData


class ResShiftSampler:
    def __init__(
            self,
            configs,
            sf=4,
            use_amp=True,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            padding_offset=16,
            seed=10000,
            ):
        self.configs = configs
        self.sf = sf
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = padding_offset
        self.rank = 0

        self.setup_seed()
        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def write_log(self, log_str):
        print(log_str, flush=True)

    def build_model(self):
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            util_net.reload_model(model, ckpt['state_dict'])
        else:
            util_net.reload_model(model, ckpt)
        self.freeze_model(model)
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder.params.get("lora_tune_decoder", False):
            lora_vae_state = ckpt['lora_vae']
        elif self.configs.autoencoder.get("tune_decoder", False):
            vae_state = ckpt['vae']
        if self.configs.autoencoder is not None:
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            if self.configs.autoencoder.params.get("lora_tune_decoder", False):
                ckpt_path = self.configs.autoencoder.ckpt_path
                self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
                self.load_model_lora(autoencoder, ckpt_path, tag='autoencoder')
                autoencoder.load_state_dict(lora_vae_state, strict=False)
            elif self.configs.autoencoder.get("tune_decoder", False):
                ckpt_path = self.configs.autoencoder.ckpt_path
                self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
                self.load_model(autoencoder, ckpt_path)
                ckpt_path =self.configs.model.ckpt_path
                self.write_log(f'Loading Finetuned decoder from {ckpt_path}...')
                autoencoder.load_state_dict(vae_state, strict=False)
            else:
                ckpt_path = self.configs.autoencoder.ckpt_path
                self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
                self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model_lora(self, model, ckpt_path=None, tag='model'):
        self.write_log(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        num_success = 0
        for key, value in model.named_parameters():
            if key in ckpt:
                value.data.copy_(ckpt[key])
                num_success += 1
            else:
                key_parts = key.split('.')
                if 'conv' in key_parts:
                    key_parts.remove('conv')
                new_key = '.'.join(key_parts)
                if new_key in ckpt:
                    value.data.copy_(ckpt[new_key])
                    num_success += 1
        assert num_success == len(ckpt)

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def sample_func(self, y0, noise_repeat=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        offset = self.padding_offset
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % offset == 0 and ori_w % offset == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / offset)) * offset - ori_h
            pad_w = (math.ceil(ori_w / offset)) * offset - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        if self.configs.model.params.cond_lq:
            model_kwargs={'lq':y0,}
        else:
            model_kwargs = None

        results = self.base_diffusion.p_sample_loop(
                y=y0,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                )    # This has included the decoding for latent space

        if flag_pad:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]

        return results.clamp_(-1.0, 1.0)

    def inference(self, in_path, out_path, in_gt_path, bs=1, noise_repeat=False):

        def _process_per_image(im_lq_tensor):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [-1, 1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.sf,
                        extra_bs=self.chop_bs,
                        )
                for im_lq_pch, index_infos in im_spliter:
                    with context():
                        im_sr_pch = self.sample_func(
                                im_lq_pch,
                                noise_repeat=noise_repeat,
                                )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                # print(im_lq_tensor.shape)
                with context():
                    im_sr_tensor = self.sample_func(
                            im_lq_tensor,
                            noise_repeat=noise_repeat
                            )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        in_gt_path = Path(in_gt_path) if not isinstance(in_gt_path, Path) else in_gt_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        assert in_path.exists()
        if not out_path.exists():
            out_path.mkdir(parents=True)

        data_config = {'type': 'base',
                        'params': {'dir_path': str(in_path),
                                    'transform_type': 'default',
                                    'transform_kwargs': {
                                        'mean': 0.5,
                                        'std': 0.5,
                                        },
                                    'need_path': True,
                                    'recursive': True,
                                    'length': None,
                                    }
                        }

        dataset = BaseData(**data_config["params"])
        self.write_log(f'Found {len(dataset)} images in {in_path}')
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                shuffle=False,
                drop_last=False,
                )

        glob_ssim = 0
        glob_psnr = 0

        for data in dataloader:
            micro_batchsize = math.ceil(bs)
            ind_start = self.rank * micro_batchsize
            ind_end = ind_start + micro_batchsize
            micro_data = {key:value[ind_start:ind_end] for key,value in data.items()}

            # nacti GT jako tensor
            img = Image.open(path.join(in_gt_path, path.basename(data["path"][0]))).convert("RGB")
            transform = T.Compose([
                T.ToTensor(),
            ])
            hr_tensor = transform(img)
            hr_tensor = hr_tensor.unsqueeze(0)

            if micro_data['lq'].shape[0] > 0:
                results = _process_per_image(
                        micro_data['lq'].cuda()
                        )    # b x h x w x c, [0, 1], RGB

                for jj in range(results.shape[0]):
                    ssim = batch_SSIM(results[jj].unsqueeze(0), hr_tensor)
                    psnr = batch_PSNR(results[jj].unsqueeze(0), hr_tensor)
                    print("file: ", micro_data['path'][jj])
                    print("SSIM: ", ssim)
                    print("PSNR: ", psnr)
                    print("=====")

                    glob_ssim += ssim
                    glob_psnr += psnr

                    im_sr = util_image.tensor2img(
                        results[jj], 
                        rgb2bgr=True, 
                        min_max=(0.0, 1.0),
                    )

                    # DEBUG
                    im_hr = util_image.tensor2img(
                        hr_tensor, 
                        rgb2bgr=True, 
                        min_max=(0.0, 1.0),
                    )
                    im_name = Path(micro_data['path'][jj]).stem
                    im_path = out_path / f"{im_name}_HR.png"
                    util_image.imwrite(im_hr, im_path, chn='bgr', dtype_in='uint8')                    

                    # zapis vysledny SR obrazek
                    im_name = Path(micro_data['path'][jj]).stem
                    im_path = out_path / f"{im_name}.png"
                    util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')

        print("GLOB PSNR: ", glob_psnr / len(dataloader))
        print("GLOB_SSIM: ", glob_ssim / len(dataloader))

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-ig", "--in_gt_path", type=str, default="", help="Input GT path.")
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
    configs = OmegaConf.load('conf/swinunet_realesrgan256.yaml')

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    # checkpoint
    if not configs.model.ckpt_path and configs.resume:
        configs.model.ckpt_path = configs.resume

    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64)
        elif args.chop_size == 256:
            chop_stride = (256 - 32)
        elif args.chop_size == 64:
            chop_stride = (64 - 16)
        else:
            raise ValueError("Chop size must be in [512, 256]")
    else:
        chop_stride = args.chop_stride
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    return configs, chop_stride


if __name__ == '__main__':

    args = get_parser()

    configs, chop_stride = get_configs(args)
    resshift_sampler = ResShiftSampler(
            configs,
            sf=4,
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
            args.in_gt_path,
            bs=1,  # batch size
            noise_repeat=False
            )
