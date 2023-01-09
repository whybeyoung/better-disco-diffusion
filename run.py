#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: run.py
@time: 2023/01/07
@contact: ybyang7@iflytek.com
@site:  
@software: PyCharm 

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛ 
"""

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import os, sys
from utils.gpu import check_nvidia
from consts import *
from utils import createPath
from utils.setup3rd_mod import setup_3rd_module
from advanced_settings import *

from basic_settings import init_image
from animation_settings import *
from ness_functions import *

import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import timm
from IPython import display
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import json
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from CLIP import clip
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import hashlib
from functools import partial

from IPython.display import Image as ipyimg
from numpy import asarray
from einops import rearrange, repeat
import torchvision
import time
from omegaconf import OmegaConf
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

USE_ADABINS = True

root_path = os.getcwd()
PROJECT_DIR = os.path.abspath(root_path)
setup_3rd_module(PROJECT_DIR)
from model import *
from CLIP import clip
import open_clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from resize_right import resize
import py3d_tools as p3dT
import disco_xform_utils as dxf
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# AdaBins stuff
if USE_ADABINS:
    from infer import InferenceHelper


def split_prompts(prompts):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


##
if key_frames:
    try:
        angle_series = get_inbetweens(parse_key_frames(angle))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `angle` correctly for key frames.\n"
            "Attempting to interpret `angle` as "
            f'"0: ({angle})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        angle = f"0: ({angle})"
        angle_series = get_inbetweens(parse_key_frames(angle))

    try:
        zoom_series = get_inbetweens(parse_key_frames(zoom))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `zoom` correctly for key frames.\n"
            "Attempting to interpret `zoom` as "
            f'"0: ({zoom})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        zoom = f"0: ({zoom})"
        zoom_series = get_inbetweens(parse_key_frames(zoom))

    try:
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_x` correctly for key frames.\n"
            "Attempting to interpret `translation_x` as "
            f'"0: ({translation_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_x = f"0: ({translation_x})"
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))

    try:
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_y` correctly for key frames.\n"
            "Attempting to interpret `translation_y` as "
            f'"0: ({translation_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_y = f"0: ({translation_y})"
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))

    try:
        translation_z_series = get_inbetweens(parse_key_frames(translation_z))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_z` correctly for key frames.\n"
            "Attempting to interpret `translation_z` as "
            f'"0: ({translation_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_z = f"0: ({translation_z})"
        translation_z_series = get_inbetweens(parse_key_frames(translation_z))

    try:
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_x` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_x` as "
            f'"0: ({rotation_3d_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_x = f"0: ({rotation_3d_x})"
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x))

    try:
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_y` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_y` as "
            f'"0: ({rotation_3d_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_y = f"0: ({rotation_3d_y})"
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y))

    try:
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_z` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_z` as "
            f'"0: ({rotation_3d_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_z = f"0: ({rotation_3d_z})"
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z))

else:
    angle = float(angle)
    zoom = float(zoom)
    translation_x = float(translation_x)
    translation_y = float(translation_y)
    translation_z = float(translation_z)
    rotation_3d_x = float(rotation_3d_x)
    rotation_3d_y = float(rotation_3d_y)
    rotation_3d_z = float(rotation_3d_z)


class DiscoDiffusion():
    def __init__(self, root_path, device, normalize, lpips_model, clip_models, steps=steps, angle=angle,
                 init_image=init_image,
                 image_name='image.png',
                 translation_x=translation_x,
                 translation_y=translation_y, init_scale=init_scale, skip_steps=skip_steps, zoom=zoom):
        self.image_name = image_name
        self.is_colab = False
        self.google_drive = False
        self.save_models_to_google_drive = False
        self.root_path = root_path
        self.initDirPath = f'{root_path}/init_images'
        self.outDirPath = f'{root_path}/images_out'
        self.model_path = f'{root_path}/models'
        # Make folder for batch
        self.batchFolder = f'{self.outDirPath}/{batch_name}'
        self.partialFolder = f'{self.batchFolder}/partials'
        self.useCPU = False
        self.secondary_model = SecondaryDiffusionImageNet2()
        self.secondary_model.load_state_dict(
            torch.load(f'{self.model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
        self.secondary_model.eval().requires_grad_(False).to(device)
        # 这啥玩意？
        self.cutout_debug = False
        self.padargs = {}
        self.normalize = normalize
        self.lpips_model = lpips_model
        # 初始化一些目录
        self.create_dirs()
        self.setup_env()
        self.setup_some_args(angle, init_image, translation_x, translation_y, init_scale, skip_steps, zoom)
        # seed
        if set_seed == 'random_seed':
            random.seed()
            self.seed = random.randint(0, 2 ** 32)
            # print(f'Using seed: {seed}')
        else:
            self.seed = int(set_seed)
        self.clip_models = clip_models
        ## gpu set
        self.device = device
        if not self.useCPU:
            if torch.cuda.get_device_capability(self.device) == (8, 0):  ## A100 fix thanks to Emad
                print('Disabling CUDNN for A100 gpu', file=sys.stderr)
                torch.backends.cudnn.enabled = False
        # Initialize MiDaS depth model.
        # It remains resident in VRAM and likely takes around 2GB VRAM.
        # You could instead initialize it for each frame (and free it after each frame) to save VRAM.. but initializing it is slow.
        self.default_models = {
            "midas_v21_small": f"{self.model_path}/midas_v21_small-70d6b9c8.pt",
            "midas_v21": f"{self.model_path}/midas_v21-f6b98070.pt",
            "dpt_large": f"{self.model_path}/dpt_large-midas-2f21e586.pt",
            "dpt_hybrid": f"{self.model_path}/dpt_hybrid-midas-501f0c75.pt",
            "dpt_hybrid_nyu": f"{self.model_path}/dpt_hybrid_nyu-2ce69ec7.pt", }

    def setup_some_args(self, angle, init_image, translation_x, translation_y, init_scale, skip_steps, zoom):
        self.angle = angle
        self.init_image = init_image
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.init_scale = init_scale
        self.skip_steps = skip_steps
        self.zoom = zoom
        pass

    def init_midas_depth_model(self, midas_model_type="dpt_large", optimize=True):
        midas_model = None
        net_w = None
        net_h = None
        resize_mode = None
        normalization = None

        print(f"Initializing MiDaS '{midas_model_type}' depth model...")
        # load network
        midas_model_path = self.default_models[midas_model_type]

        if midas_model_type == "dpt_large":  # DPT-Large
            midas_model = DPTDepthModel(
                path=midas_model_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif midas_model_type == "dpt_hybrid":  # DPT-Hybrid
            midas_model = DPTDepthModel(
                path=midas_model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif midas_model_type == "dpt_hybrid_nyu":  # DPT-Hybrid-NYU
            midas_model = DPTDepthModel(
                path=midas_model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif midas_model_type == "midas_v21":
            midas_model = MidasNet(midas_model_path, non_negative=True)
            net_w, net_h = 384, 384
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif midas_model_type == "midas_v21_small":
            midas_model = MidasNet_small(midas_model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                                         non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            print(f"midas_model_type '{midas_model_type}' not implemented")
            assert False

        midas_transform = T.Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        midas_model.eval()

        if optimize == True:
            if self.device == torch.device("cuda"):
                midas_model = midas_model.to(memory_format=torch.channels_last)
                midas_model = midas_model.half()

        midas_model.to(self.device)

        print(f"MiDaS '{midas_model_type}' depth model initialized.")
        return midas_model, midas_transform, net_w, net_h, resize_mode, normalization

    def setup_output_image(self, name):
        self.image_name = name

    def create_dirs(self):
        createPath(self.initDirPath)
        createPath(self.outDirPath)
        createPath(self.model_path)
        createPath(f'{self.root_path}/pretrained')
        createPath(self.batchFolder)
        createPath(self.partialFolder)

    def setup_env(self):
        # If running locally, there's a good chance your env will need this in order to not crash upon np.matmul() or similar operations.
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    def do_run(self, args, ):
        image_name = self.image_name
        check_nvidia()
        start_frame = 0
        batchNum = len(glob(self.batchFolder + "/*.txt"))
        while os.path.isfile(f"{self.batchFolder}/{batch_name}({batchNum})_settings.txt") or os.path.isfile(
                f"{self.batchFolder}/{batch_name}-{batchNum}_settings.txt"):
            batchNum += 1

        print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')
        self.batchNum = batchNum
        args['batchNum'] = batchNum
        args['start_frame'] = start_frame

        args = SimpleNamespace(**args)
        self.args = args
        self.steps = args.steps
        self.eta = args.eta
        print('Prepping model...')
        model, diffusion = create_model_and_diffusion(**self.model_config)
        if diffusion_model == 'custom':
            model.load_state_dict(torch.load(custom_path, map_location='cpu'))
        else:
            m = f'{self.model_path}/{get_model_filename(diffusion_model)}'
            print("loading %s" % m)
            model.load_state_dict(
                torch.load(m, map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)
        for name, param in model.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                param.requires_grad_()
        if self.model_config['use_fp16']:
            model.convert_to_fp16()

        gc.collect()
        torch.cuda.empty_cache()

        seed = args.seed
        print(range(args.start_frame, args.max_frames))

        for frame_num in range(args.start_frame, args.max_frames):
            if stop_on_next_loop:
                break

            display.clear_output(wait=True)

            # Print Frame progress if animation mode is on
            if args.animation_mode != "None":
                batchBar = tqdm(range(args.max_frames), desc="Frames")
                batchBar.n = frame_num
                batchBar.refresh()

            # Inits if not video frames
            if args.animation_mode != "Video Input":
                if args.init_image in ['', 'none', 'None', 'NONE']:
                    init_image = None
                else:
                    init_image = args.init_image
                init_scale = args.init_scale
                skip_steps = args.skip_steps

            if args.animation_mode == "2D":
                if args.key_frames:
                    angle = args.angle_series[frame_num]
                    zoom = args.zoom_series[frame_num]
                    translation_x = args.translation_x_series[frame_num]
                    translation_y = args.translation_y_series[frame_num]
                    print(
                        f'angle: {angle}',
                        f'zoom: {zoom}',
                        f'translation_x: {translation_x}',
                        f'translation_y: {translation_y}',
                    )

                if frame_num > 0:
                    seed += 1
                    if args.resume_run and frame_num == start_frame:
                        img_0 = cv2.imread(self.batchFolder + f"/{batch_name}({batchNum})_{start_frame - 1:04}.png")
                    else:
                        img_0 = cv2.imread('prevFrame.png')
                    center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
                    trans_mat = np.float32(
                        [[1, 0, translation_x],
                         [0, 1, translation_y]]
                    )
                    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
                    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
                    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
                    transformation_matrix = np.matmul(rot_mat, trans_mat)
                    img_0 = cv2.warpPerspective(
                        img_0,
                        transformation_matrix,
                        (img_0.shape[1], img_0.shape[0]),
                        borderMode=cv2.BORDER_WRAP
                    )

                    cv2.imwrite('prevFrameScaled.png', img_0)
                    init_image = 'prevFrameScaled.png'
                    init_scale = args.frames_scale
                    skip_steps = args.calc_frames_skip_steps

            loss_values = []

            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True

            target_embeds, weights = [], []

            if args.prompts_series is not None and frame_num >= len(args.prompts_series):
                frame_prompt = args.prompts_series[-1]
            elif args.prompts_series is not None:
                frame_prompt = args.prompts_series[frame_num]
            else:
                frame_prompt = []

            print(args.image_prompts_series)
            if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
                image_prompt = args.image_prompts_series[-1]
            elif args.image_prompts_series is not None:
                image_prompt = args.image_prompts_series[frame_num]
            else:
                image_prompt = []

            print(f'Frame {frame_num} Prompt: {frame_prompt}')

            model_stats = []
            for clip_model in self.clip_models:
                cutn = 16
                model_stat = {"clip_model": None, "target_embeds": [], "make_cutouts": None, "weights": []}
                model_stat["clip_model"] = clip_model

                for prompt in frame_prompt:
                    txt, weight = parse_prompt(prompt)
                    txt = clip_model.encode_text(clip.tokenize(prompt).to(self.device)).float()

                    if args.fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append(
                                (txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0, 1))
                            model_stat["weights"].append(weight)
                    else:
                        model_stat["target_embeds"].append(txt)
                        model_stat["weights"].append(weight)

                if image_prompt:
                    model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn,
                                                             skip_augs=skip_augs)
                    for prompt in image_prompt:
                        path, weight = parse_prompt(prompt)
                        img = Image.open(fetch(path)).convert('RGB')
                        img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
                        batch = model_stat["make_cutouts"](TF.to_tensor(img).to(self.device).unsqueeze(0).mul(2).sub(1))
                        embed = clip_model.encode_image(self.normalize(batch)).float()
                        if fuzzy_prompt:
                            for i in range(25):
                                model_stat["target_embeds"].append(
                                    (embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0, 1))
                                weights.extend([weight / cutn] * cutn)
                        else:
                            model_stat["target_embeds"].append(embed)
                            model_stat["weights"].extend([weight / cutn] * cutn)

                model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
                model_stat["weights"] = torch.tensor(model_stat["weights"], device=self.device)
                if model_stat["weights"].sum().abs() < 1e-3:
                    raise RuntimeError('The weights must not sum to 0.')
                model_stat["weights"] /= model_stat["weights"].sum().abs()
                model_stats.append(model_stat)

            init = None
            if init_image is not None:
                init = Image.open(fetch(init_image)).convert('RGB')
                init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
                init = TF.to_tensor(init).to(self.device).unsqueeze(0).mul(2).sub(1)

            if args.perlin_init:
                if args.perlin_mode == 'color':
                    init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False, self.device)
                    init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, False, self.device)
                elif args.perlin_mode == 'gray':
                    init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, True, self.device)
                    init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True, self.device)
                else:
                    init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False, self.device)
                    init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True, self.device)
                # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
                init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(self.device).unsqueeze(0).mul(2).sub(1)
                del init2

            cur_t = None

            def cond_fn(x, t, y=None):
                with torch.enable_grad():
                    x_is_NaN = False
                    x = x.detach().requires_grad_()
                    n = x.shape[0]
                    if use_secondary_model is True:
                        alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=self.device,
                                             dtype=torch.float32)
                        sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=self.device,
                                             dtype=torch.float32)
                        cosine_t = alpha_sigma_to_t(alpha, sigma)
                        out = self.secondary_model(x, cosine_t[None].repeat([n])).pred
                        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                        x_in = out * fac + x * (1 - fac)
                        x_in_grad = torch.zeros_like(x_in)
                    else:
                        my_t = torch.ones([n], device=self.device, dtype=torch.long) * cur_t
                        out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                        x_in = out['pred_xstart'] * fac + x * (1 - fac)
                        x_in_grad = torch.zeros_like(x_in)
                    for model_stat in model_stats:
                        for i in range(args.cutn_batches):
                            t_int = int(t.item()) + 1  # errors on last step without +1, need to find source
                            # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                            try:
                                input_resolution = model_stat["clip_model"].visual.input_resolution
                            except:
                                input_resolution = 224

                            cuts = MakeCutoutsDango(args, self.padargs, input_resolution,
                                                    Overview=args.cut_overview[1000 - t_int],
                                                    InnerCrop=args.cut_innercut[1000 - t_int],
                                                    IC_Size_Pow=args.cut_ic_pow[1000 - t_int],
                                                    IC_Grey_P=args.cut_icgray_p[1000 - t_int],
                                                    cutout_debug=self.cutout_debug,
                                                    )
                            clip_in = self.normalize(cuts(x_in.add(1).div(2)))
                            image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                            dists = spherical_dist_loss(image_embeds.unsqueeze(1),
                                                        model_stat["target_embeds"].unsqueeze(0))
                            dists = dists.view(
                                [args.cut_overview[1000 - t_int] + args.cut_innercut[1000 - t_int], n, -1])
                            losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                            loss_values.append(losses.sum().item())  # log loss, probably shouldn't do per cutn_batch
                            x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
                    tv_losses = tv_loss(x_in)
                    if use_secondary_model is True:
                        range_losses = range_loss(out)
                    else:
                        range_losses = range_loss(out['pred_xstart'])
                    sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                    loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
                    if init is not None and init_scale:
                        init_losses = self.lpips_model(x_in, init)
                        loss = loss + init_losses.sum() * init_scale
                    x_in_grad += torch.autograd.grad(loss, x_in)[0]
                    if torch.isnan(x_in_grad).any() == False:
                        grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                    else:
                        # print("NaN'd")
                        x_is_NaN = True
                        grad = torch.zeros_like(x)
                if args.clamp_grad and x_is_NaN == False:
                    magnitude = grad.square().mean().sqrt()
                    return grad * magnitude.clamp(max=args.clamp_max) / magnitude  # min=-0.02, min=-clamp_max,
                return grad

            if args.diffusion_sampling_mode == 'ddim':
                sample_fn = diffusion.ddim_sample_loop_progressive
            else:
                sample_fn = diffusion.plms_sample_loop_progressive

            image_display = Output()
            for i in range(args.n_batches):
                if args.animation_mode == 'None':
                    display.clear_output(wait=True)
                    batchBar = tqdm(range(args.n_batches), desc="Batches")
                    batchBar.n = i
                    batchBar.refresh()
                print('')
                display.display(image_display)
                gc.collect()
                torch.cuda.empty_cache()
                cur_t = diffusion.num_timesteps - skip_steps - 1
                total_steps = cur_t

                if perlin_init:
                    init = regen_perlin(args.batch_size, self.device)

                if args.diffusion_sampling_mode == 'ddim':
                    samples = sample_fn(
                        model,
                        (args.batch_size, 3, args.side_y, args.side_x),
                        clip_denoised=clip_denoised,
                        model_kwargs={},
                        cond_fn=cond_fn,
                        progress=True,
                        skip_timesteps=skip_steps,
                        init_image=init,
                        randomize_class=randomize_class,
                        eta=args.eta,
                        transformation_fn=symmetry_transformation_fn,
                        transformation_percent=args.transformation_percent
                    )
                else:
                    samples = sample_fn(
                        model,
                        (args.batch_size, 3, args.side_y, args.side_x),
                        clip_denoised=clip_denoised,
                        model_kwargs={},
                        cond_fn=cond_fn,
                        progress=True,
                        skip_timesteps=skip_steps,
                        init_image=init,
                        randomize_class=randomize_class,
                        order=2,
                    )

                # with run_display:
                # display.clear_output(wait=True)
                for j, sample in enumerate(samples):
                    cur_t -= 1
                    intermediateStep = False
                    if args.steps_per_checkpoint is not None:
                        if j % steps_per_checkpoint == 0 and j > 0:
                            intermediateStep = True
                    elif j in args.intermediate_saves:
                        intermediateStep = True
                    with image_display:
                        if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                            for k, image in enumerate(sample['pred_xstart']):
                                # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                                current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                                percent = math.ceil(j / total_steps * 100)
                                if args.n_batches > 0:
                                    # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                                    if cur_t == -1 and args.intermediates_in_subfolder is True:
                                        save_num = f'{frame_num:04}' if animation_mode != "None" else i
                                        filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                                    else:
                                        # If we're working with percentages, append it
                                        if args.steps_per_checkpoint is not None:
                                            filename = f'{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png'
                                        # Or else, iIf we're working with specific steps, append those
                                        else:
                                            filename = f'{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png'
                                image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                                if j % args.display_rate == 0 or cur_t == -1:
                                    image.save(image_name)
                                    display.clear_output(wait=True)
                                    display.display(display.Image(image_name))
                                if args.steps_per_checkpoint is not None:
                                    if j % args.steps_per_checkpoint == 0 and j > 0:
                                        if args.intermediates_in_subfolder is True:
                                            image.save(f'{self.partialFolder}/{filename}')
                                        else:
                                            image.save(f'{self.batchFolder}/{filename}')
                                else:
                                    if j in args.intermediate_saves:
                                        if args.intermediates_in_subfolder is True:
                                            image.save(f'{self.partialFolder}/{filename}')
                                        else:
                                            image.save(f'{self.batchFolder}/{filename}')
                                if cur_t == -1:
                                    if frame_num == 0:
                                        self.save_settings()
                                    if args.animation_mode != "None":
                                        image.save('prevFrame.png')
                                    image.save(f'{self.batchFolder}/{filename}')
                                    if args.animation_mode == "3D":
                                        # If turbo, save a blended image
                                        if turbo_mode and frame_num > 0:
                                            # Mix new image with prevFrameScaled
                                            blend_factor = (1) / int(turbo_steps)
                                            newFrame = cv2.imread('prevFrame.png')  # This is already updated..
                                            prev_frame_warped = cv2.imread('prevFrameScaled.png')
                                            blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped,
                                                                           (1 - blend_factor), 0.0)
                                            cv2.imwrite(f'{self.batchFolder}/{filename}', blendedImage)
                                        else:
                                            image.save(f'{self.batchFolder}/{filename}')

                                    # if frame_num != args.max_frames-1:
                                    #   display.clear_output()

                plt.plot(np.array(loss_values), 'r')

    def save_settings(self):
        setting_list = {
            'text_prompts': self.args.text_prompts,
            'image_prompts': self.args.image_prompts,
            'clip_guidance_scale': clip_guidance_scale,
            'tv_scale': tv_scale,
            'range_scale': range_scale,
            'sat_scale': sat_scale,
            # 'cutn': cutn,
            'cutn_batches': cutn_batches,
            'max_frames': max_frames,
            'interp_spline': interp_spline,
            # 'rotation_per_frame': rotation_per_frame,
            'init_image': init_image,
            'init_scale': init_scale,
            'skip_steps': skip_steps,
            # 'zoom_per_frame': zoom_per_frame,
            'frames_scale': frames_scale,
            'frames_skip_steps': frames_skip_steps,
            'perlin_init': perlin_init,
            'perlin_mode': perlin_mode,
            'skip_augs': skip_augs,
            'randomize_class': randomize_class,
            'clip_denoised': clip_denoised,
            'clamp_grad': clamp_grad,
            'clamp_max': clamp_max,
            'seed': self.args.seed,
            'fuzzy_prompt': fuzzy_prompt,
            'rand_mag': rand_mag,
            'eta': self.eta,
            'width': width_height[0],
            'height': width_height[1],
            'diffusion_model': diffusion_model,
            'use_secondary_model': use_secondary_model,
            'steps': self.steps,
            'diffusion_steps': self.diffusion_steps,
            'diffusion_sampling_mode': diffusion_sampling_mode,
            'ViTB32': ViTB32,
            'ViTB16': ViTB16,
            'ViTL14': ViTL14,
            'ViTL14_336px': ViTL14_336px,
            'RN101': RN101,
            'RN50': RN50,
            'RN50x4': RN50x4,
            'RN50x16': RN50x16,
            'RN50x64': RN50x64,
            'ViTB32_laion2b_e16': ViTB32_laion2b_e16,
            'ViTB32_laion400m_e31': ViTB32_laion400m_e31,
            'ViTB32_laion400m_32': ViTB32_laion400m_32,
            'ViTB32quickgelu_laion400m_e31': ViTB32quickgelu_laion400m_e31,
            'ViTB32quickgelu_laion400m_e32': ViTB32quickgelu_laion400m_e32,
            'ViTB16_laion400m_e31': ViTB16_laion400m_e31,
            'ViTB16_laion400m_e32': ViTB16_laion400m_e32,
            'RN50_yffcc15m': RN50_yffcc15m,
            'RN50_cc12m': RN50_cc12m,
            'RN50_quickgelu_yfcc15m': RN50_quickgelu_yfcc15m,
            'RN50_quickgelu_cc12m': RN50_quickgelu_cc12m,
            'RN101_yfcc15m': RN101_yfcc15m,
            'RN101_quickgelu_yfcc15m': RN101_quickgelu_yfcc15m,
            'cut_overview': str(cut_overview),
            'cut_innercut': str(cut_innercut),
            'cut_ic_pow': str(cut_ic_pow),
            'cut_icgray_p': str(cut_icgray_p),
            'key_frames': key_frames,
            'max_frames': max_frames,
            'angle': angle,
            'zoom': zoom,
            'translation_x': translation_x,
            'translation_y': translation_y,
            'translation_z': translation_z,
            'rotation_3d_x': rotation_3d_x,
            'rotation_3d_y': rotation_3d_y,
            'rotation_3d_z': rotation_3d_z,
            'midas_depth_model': midas_depth_model,
            'midas_weight': midas_weight,
            'near_plane': near_plane,
            'far_plane': far_plane,
            'fov': fov,
            'padding_mode': padding_mode,
            'sampling_mode': sampling_mode,
            'video_init_path': video_init_path,
            'extract_nth_frame': extract_nth_frame,
            'video_init_seed_continuity': video_init_seed_continuity,
            'turbo_mode': turbo_mode,
            'turbo_steps': turbo_steps,
            'turbo_preroll': turbo_preroll,
            'use_horizontal_symmetry': use_horizontal_symmetry,
            'use_vertical_symmetry': use_vertical_symmetry,
            'transformation_percent': transformation_percent,
            # video init settings
            'video_init_steps': video_init_steps,
            'video_init_clip_guidance_scale': video_init_clip_guidance_scale,
            'video_init_tv_scale': video_init_tv_scale,
            'video_init_range_scale': video_init_range_scale,
            'video_init_sat_scale': video_init_sat_scale,
            'video_init_cutn_batches': video_init_cutn_batches,
            'video_init_skip_steps': video_init_skip_steps,
            'video_init_frames_scale': video_init_frames_scale,
            'video_init_frames_skip_steps': video_init_frames_skip_steps,
            # warp settings
            'video_init_flow_warp': video_init_flow_warp,
            'video_init_flow_blend': video_init_flow_blend,
            'video_init_check_consistency': video_init_check_consistency,
            'video_init_blend_mode': video_init_blend_mode
        }
        # print('Settings:', setting_list)
        with open(f"{self.batchFolder}/{batch_name}({self.batchNum})_settings.txt", "w+",
                  encoding="utf-8") as f:  # save settings
            json.dump(setting_list, f, ensure_ascii=False, indent=4)

    def setup_model_config(self, timestep_respacing, diffusion_steps):
        # model config
        model_config = model_and_diffusion_defaults()
        if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
            model_config.update({
                'attention_resolutions': '32, 16, 8',
                'class_cond': False,
                'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
                'rescale_timesteps': True,
                'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
                'image_size': 512,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_checkpoint': use_checkpoint,
                'use_fp16': not self.useCPU,
                'use_scale_shift_norm': True,
            })
        elif diffusion_model == '256x256_diffusion_uncond':
            model_config.update({
                'attention_resolutions': '32, 16, 8',
                'class_cond': False,
                'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
                'rescale_timesteps': True,
                'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_checkpoint': use_checkpoint,
                'use_fp16': not self.useCPU,
                'use_scale_shift_norm': True,
            })
        elif diffusion_model == 'portrait_generator_v001':
            model_config.update({
                'attention_resolutions': '32, 16, 8',
                'class_cond': False,
                'diffusion_steps': 1000,
                'rescale_timesteps': True,
                'image_size': 512,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 128,
                'num_heads': 4,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_checkpoint': use_checkpoint,
                'use_fp16': True,
                'use_scale_shift_norm': True,
            })
        else:  # E.g. A model finetuned by KaliYuga
            model_config.update({
                'attention_resolutions': '16',
                'class_cond': False,
                'diffusion_steps': 1000,
                'rescale_timesteps': True,
                'timestep_respacing': 'ddim100',
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 128,
                'num_heads': 1,
                'num_res_blocks': 2,
                'use_checkpoint': use_checkpoint,
                'use_fp16': True,
                'use_scale_shift_norm': False,
            })

        model_default = model_config['image_size']
        model_config.update({
            'timestep_respacing': timestep_respacing,
            'diffusion_steps': diffusion_steps,
        })
        self.timestep_respacing = timestep_respacing
        self.diffusion_steps = diffusion_steps
        self.model_config = model_config


# @title 1.5 Define necessary functions
def interp(t):
    return 3 * t ** 2 - 2 * t ** 3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=None):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True, device='cpu'):
    out = perlin_ms(octaves, width, height, grayscale, device)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out


def regen_perlin(batch_size=1, device='cpu'):
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False, device)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, False, device)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, True, device)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True, device)
    else:
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False, device)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True, device)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)


def _build_args(text_prompts, image_prompts, seed, batch_size, steps, display_rate, batch_name, width_height, tv_scale,
                range_scale,
                sat_scale, cutn_batches, init_image, init_scale, skip_steps, side_x, side_y, timestep_respacing,
                diffusion_steps, resume_run, eta):
    return {
        'eta': eta,
        'resume_run': resume_run,
        'prompts_series': split_prompts(text_prompts) if text_prompts else None,
        'image_prompts_series': split_prompts(image_prompts) if image_prompts else None,
        'seed': seed,
        'display_rate': display_rate,
        'n_batches': n_batches if animation_mode == 'None' else 1,
        'batch_size': batch_size,
        'batch_name': batch_name,
        'steps': steps,
        'diffusion_sampling_mode': diffusion_sampling_mode,
        'width_height': width_height,
        'clip_guidance_scale': clip_guidance_scale,
        'tv_scale': tv_scale,
        'range_scale': range_scale,
        'sat_scale': sat_scale,
        'cutn_batches': cutn_batches,
        'init_image': init_image,
        'init_scale': init_scale,
        'skip_steps': skip_steps,
        'side_x': side_x,
        'side_y': side_y,
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
        'animation_mode': animation_mode,
        'video_init_path': video_init_path,
        'extract_nth_frame': extract_nth_frame,
        'video_init_seed_continuity': video_init_seed_continuity,
        'key_frames': key_frames,
        'max_frames': max_frames if animation_mode != "None" else 1,
        'interp_spline': interp_spline,
        'angle': angle,
        'zoom': zoom,
        'translation_x': translation_x,
        'translation_y': translation_y,
        'translation_z': translation_z,
        'rotation_3d_x': rotation_3d_x,
        'rotation_3d_y': rotation_3d_y,
        'rotation_3d_z': rotation_3d_z,
        'midas_depth_model': midas_depth_model,
        'midas_weight': midas_weight,
        'near_plane': near_plane,
        'far_plane': far_plane,
        'fov': fov,
        'padding_mode': padding_mode,
        'sampling_mode': sampling_mode,
        'angle_series': angle_series,
        'zoom_series': zoom_series,
        'translation_x_series': translation_x_series,
        'translation_y_series': translation_y_series,
        'translation_z_series': translation_z_series,
        'rotation_3d_x_series': rotation_3d_x_series,
        'rotation_3d_y_series': rotation_3d_y_series,
        'rotation_3d_z_series': rotation_3d_z_series,
        'frames_scale': frames_scale,
        'skip_step_ratio': skip_step_ratio,
        'calc_frames_skip_steps': math.floor(steps * skip_step_ratio)
        ,
        'text_prompts': text_prompts,
        'image_prompts': image_prompts,
        'cut_overview': eval(cut_overview),
        'cut_innercut': eval(cut_innercut),
        'cut_ic_pow': eval(cut_ic_pow),
        'cut_icgray_p': eval(cut_icgray_p),
        'intermediate_saves': intermediate_saves,
        'intermediates_in_subfolder': intermediates_in_subfolder,
        'steps_per_checkpoint': steps_per_checkpoint,
        'perlin_init': perlin_init,
        'perlin_mode': perlin_mode,
        'set_seed': set_seed,
        'eta': eta,
        'clamp_grad': clamp_grad,
        'clamp_max': clamp_max,
        'skip_augs': skip_augs,
        'randomize_class': randomize_class,
        'clip_denoised': clip_denoised,
        'fuzzy_prompt': fuzzy_prompt,
        'rand_mag': rand_mag,
        'turbo_mode': turbo_mode,
        'turbo_steps': turbo_steps,
        'turbo_preroll': turbo_preroll,
        'use_vertical_symmetry': use_vertical_symmetry,
        'use_horizontal_symmetry': use_horizontal_symmetry,
        'transformation_percent': transformation_percent,
        # video init settings
        'video_init_steps': video_init_steps,
        'video_init_clip_guidance_scale': video_init_clip_guidance_scale,
        'video_init_tv_scale': video_init_tv_scale,
        'video_init_range_scale': video_init_range_scale,
        'video_init_sat_scale': video_init_sat_scale,
        'video_init_cutn_batches': video_init_cutn_batches,
        'video_init_skip_steps': video_init_skip_steps,
        'video_init_frames_scale': video_init_frames_scale,
        'video_init_frames_skip_steps': video_init_frames_skip_steps,
        # warp settings
        'video_init_flow_warp': video_init_flow_warp,
        'video_init_flow_blend': video_init_flow_blend,
        'video_init_check_consistency': video_init_check_consistency,
        'video_init_blend_mode': video_init_blend_mode
    }


class DDRunner():
    def __init__(self, text_prompts, batch_size=1, steps=steps):
        self.text_prompts = text_prompts
        self.batch_size = batch_size
        self.steps = steps
        MAX_ADABINS_AREA = 500000
        device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        print('Using device:', device)
        self.resume_run = False  # @param{type: 'boolean'}
        run_to_resume = 'latest'  # @param{type: 'string'}
        resume_from_frame = 'latest'  # @param{type: 'string'}
        retain_overwritten_frames = False  # @param{type: 'boolean'}

        self.image_prompts = {
            # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
        }

        self.display_rate = 20  # @param{type: 'number'}

        # clip set
        clip_models = []
        if ViTB32: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
        if ViTB16: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device))
        if ViTL14: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device))
        if ViTL14_336px: clip_models.append(
            clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device))
        if RN50: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
        if RN50x4: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device))
        if RN50x16: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device))
        if RN50x64: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device))
        if RN101: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device))
        if ViTB32_laion2b_e16: clip_models.append(
            open_clip.create_model('ViT-B-32', pretrained='laion2b_e16').eval().requires_grad_(False).to(device))
        if ViTB32_laion400m_e31: clip_models.append(
            open_clip.create_model('ViT-B-32', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
        if ViTB32_laion400m_32: clip_models.append(
            open_clip.create_model('ViT-B-32', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
        if ViTB32quickgelu_laion400m_e31: clip_models.append(
            open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e31').eval().requires_grad_(False).to(
                device))
        if ViTB32quickgelu_laion400m_e32: clip_models.append(
            open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e32').eval().requires_grad_(False).to(
                device))
        if ViTB16_laion400m_e31: clip_models.append(
            open_clip.create_model('ViT-B-16', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
        if ViTB16_laion400m_e32: clip_models.append(
            open_clip.create_model('ViT-B-16', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
        if RN50_yffcc15m: clip_models.append(
            open_clip.create_model('RN50', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
        if RN50_cc12m: clip_models.append(
            open_clip.create_model('RN50', pretrained='cc12m').eval().requires_grad_(False).to(device))
        if RN50_quickgelu_yfcc15m: clip_models.append(
            open_clip.create_model('RN50-quickgelu', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
        if RN50_quickgelu_cc12m: clip_models.append(
            open_clip.create_model('RN50-quickgelu', pretrained='cc12m').eval().requires_grad_(False).to(device))
        if RN101_yfcc15m: clip_models.append(
            open_clip.create_model('RN101', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
        if RN101_quickgelu_yfcc15m: clip_models.append(
            open_clip.create_model('RN101-quickgelu', pretrained='yfcc15m').eval().requires_grad_(False).to(device))

        normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        lpips_model = lpips.LPIPS(net='vgg').to(device)

        # Update Model Settings
        self.timestep_respacing = f'ddim{self.steps}'
        self.diffusion_steps = (1000 // self.steps) * self.steps if self.steps < 1000 else self.steps
        print("steps : ", self.steps)
        self.dd = DiscoDiffusion(root_path, device, normalize, lpips_model, clip_models, self.steps)
        self.dd.setup_model_config(self.timestep_respacing, self.diffusion_steps)
        self.args = _build_args(self.text_prompts, self.image_prompts, self.dd.seed, self.batch_size, self.steps,
                                self.display_rate,
                                batch_name,
                                width_height,
                                tv_scale,
                                range_scale,
                                sat_scale, cutn_batches, init_image, init_scale, skip_steps, side_x, side_y,
                                self.timestep_respacing,
                                self.diffusion_steps, self.resume_run)

    def run(self, image_name="image.png"):
        self.dd.setup_output_image(image_name)
        self.dd.do_run(self.args)

    def setup_args(self, text_prompts, display_rate=20, batch_size=1, steps=steps, width_height=width_height,
                   tv_scale=tv_scale,
                   range_scale=range_scale,
                   sat_scale=sat_scale, cutn_batches=cutn_batches, init_image=init_image, init_scale=init_scale,
                   skip_steps=skip_steps, side_x=side_x,
                   side_y=side_y,
                   resume_run=False,
                   eta=eta):
        # Update Model Settings
        print("setup... , steps", steps)
        timestep_respacing = f'ddim{steps}'
        diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
        self.args = _build_args(text_prompts,
                                image_prompts=self.image_prompts,
                                seed=self.dd.seed,
                                batch_size=batch_size,
                                steps=steps,
                                display_rate=display_rate,
                                batch_name=batch_name,
                                width_height=width_height,
                                tv_scale=tv_scale,
                                range_scale=range_scale,
                                sat_scale=sat_scale,
                                cutn_batches=cutn_batches,
                                init_image=init_image,
                                init_scale=init_scale,
                                skip_steps=skip_steps,
                                side_x=side_x,
                                side_y=side_y,
                                timestep_respacing=timestep_respacing,
                                diffusion_steps=diffusion_steps,
                                resume_run=self.resume_run,
                                eta=eta)
        self.timestep_respacing = timestep_respacing
        self.diffusion_steps = diffusion_steps
        self.dd.setup_model_config(timestep_respacing, diffusion_steps)


if __name__ == '__main__':
    input_text = {
        0: ["A beautiful woman is skiing, and the flags of various countries are planted on the snow field",
            "white color scheme"],
    }
    test_text_prompts = {
        0: [
            "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.",
            "yellow color scheme"],
    }
    ds = DDRunner(input_text)
    #     def setup_args(self, text_prompts, display_rate=20, batch_size=1, steps=steps, width_height=width_height,
    #                    tv_scale=tv_scale,
    #                    range_scale=range_scale,
    #                    sat_scale=sat_scale, cutn_batches=cutn_batches, init_image=init_image, init_scale=init_scale,
    #                    skip_steps=skip_steps, side_x=side_x,
    #                    side_y=side_y, timestep_respacing=f'ddim{steps}',
    #                    diffusion_steps=(1000 // steps) * steps if steps < 1000 else steps,
    #                    resume_run=False):
    ds.setup_args(test_text_prompts, width_height=[1280, 768], steps=50)
    ds.run("image2.png")
