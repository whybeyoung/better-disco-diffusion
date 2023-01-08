#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: advanced_settings
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
from basic_settings import *
import math

perlin_init = False  #@param{type: 'boolean'}
perlin_mode = 'mixed' #@param ['mixed', 'color', 'gray']
set_seed = 'random_seed' #@param{type: 'string'}
eta = 0.8 #@param{type: 'number'}
clamp_grad = True #@param{type: 'boolean'}
clamp_max = 0.05 #@param{type: 'number'}


### EXTRA ADVANCED SETTINGS:
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05


# 2. Diffusion and CLIP model settings


diffusion_model = "512x512_diffusion_uncond_finetune_008100"
use_secondary_model = True #@param {type: 'boolean'}
diffusion_sampling_mode = 'ddim' #@param ['plms','ddim']
custom_path = '/content/drive/MyDrive/deep_learning/ddpm/ema_0.9999_058000.pt'#@param {type: 'string'}
#@markdown #####**CLIP settings:**
use_checkpoint = True #@param {type: 'boolean'}
ViTB32 = True #@param{type:"boolean"}
ViTB16 = True #@param{type:"boolean"}
ViTL14 = False #@param{type:"boolean"}
ViTL14_336px = False #@param{type:"boolean"}
RN101 = False #@param{type:"boolean"}
RN50 = True #@param{type:"boolean"}
RN50x4 = False #@param{type:"boolean"}
RN50x16 = False #@param{type:"boolean"}
RN50x64 = False #@param{type:"boolean"}

#@markdown #####**OpenCLIP settings:**
ViTB32_laion2b_e16 = False #@param{type:"boolean"}
ViTB32_laion400m_e31 = False #@param{type:"boolean"}
ViTB32_laion400m_32 = False #@param{type:"boolean"}
ViTB32quickgelu_laion400m_e31 = False #@param{type:"boolean"}
ViTB32quickgelu_laion400m_e32 = False #@param{type:"boolean"}
ViTB16_laion400m_e31 = False #@param{type:"boolean"}
ViTB16_laion400m_e32 = False #@param{type:"boolean"}
RN50_yffcc15m = False #@param{type:"boolean"}
RN50_cc12m = False #@param{type:"boolean"}
RN50_quickgelu_yfcc15m = False #@param{type:"boolean"}
RN50_quickgelu_cc12m = False #@param{type:"boolean"}
RN101_yfcc15m = False #@param{type:"boolean"}
RN101_quickgelu_yfcc15m = False #@param{type:"boolean"}

#@markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False #@param{type:"boolean"}



kaliyuga_pixel_art_model_names = ['pixelartdiffusion_expanded', 'pixel_art_diffusion_hard_256', 'pixel_art_diffusion_soft_256', 'pixelartdiffusion4k', 'PulpSciFiDiffusion']
kaliyuga_watercolor_model_names = ['watercolordiffusion', 'watercolordiffusion_2']
kaliyuga_pulpscifi_model_names = ['PulpSciFiDiffusion']
diffusion_models_256x256_list = ['256x256_diffusion_uncond'] + kaliyuga_pixel_art_model_names + kaliyuga_watercolor_model_names + kaliyuga_pulpscifi_model_names


#@markdown ####**Coherency Settings:**
#@markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
frames_scale = 1500 #@param{type: 'integer'}
#@markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
frames_skip_steps = '60%' #@param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}

#@markdown ####**Video Init Coherency Settings:**
#@markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
video_init_frames_scale = 15000 #@param{type: 'integer'}
#@markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
video_init_frames_skip_steps = '70%' #@param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}


vr_mode = False #@param {type:"boolean"}
#@markdown `vr_eye_angle` is the y-axis rotation of the eyes towards the center
vr_eye_angle = 0.5 #@param{type:"number"}
#@markdown interpupillary distance (between the eyes)
vr_ipd = 5.0 #@param{type:"number"}

skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

#@markdown ####**Transformation Settings:**
use_vertical_symmetry = False #@param {type:"boolean"}
use_horizontal_symmetry = False #@param {type:"boolean"}
transformation_percent = [0.09] #@param

#@markdown cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.

cut_overview = "[12]*400+[4]*600" #@param {type: 'string'}
cut_innercut = "[4]*400+[12]*600" #@param {type: 'string'}
cut_ic_pow = "[1]*1000" #@param {type: 'string'}
cut_icgray_p = "[0.2]*400+[0]*600" #@param {type: 'string'}

#@markdown KaliYuga model settings. Refer to [cut_ic_pow](https://ezcharts.miraheze.org/wiki/Category:Cut_ic_pow) as a guide. Values between 1 and 100 all work.
pad_or_pulp_cut_overview = "[15]*100+[15]*100+[12]*100+[12]*100+[6]*100+[4]*100+[2]*200+[0]*200" #@param {type: 'string'}
pad_or_pulp_cut_innercut = "[1]*100+[1]*100+[4]*100+[4]*100+[8]*100+[8]*100+[10]*200+[10]*200" #@param {type: 'string'}
pad_or_pulp_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300" #@param {type: 'string'}
pad_or_pulp_cut_icgray_p = "[0.87]*100+[0.78]*50+[0.73]*50+[0.64]*60+[0.56]*40+[0.50]*50+[0.33]*100+[0.19]*150+[0]*400" #@param {type: 'string'}

watercolor_cut_overview = "[14]*200+[12]*200+[4]*400+[0]*200" #@param {type: 'string'}
watercolor_cut_innercut = "[2]*200+[4]*200+[12]*400+[12]*200" #@param {type: 'string'}
watercolor_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300" #@param {type: 'string'}
watercolor_cut_icgray_p = "[0.7]*100+[0.6]*100+[0.45]*100+[0.3]*100+[0]*600" #@param {type: 'string'}

if (diffusion_model in kaliyuga_pixel_art_model_names) or (diffusion_model in kaliyuga_pulpscifi_model_names):
    cut_overview = pad_or_pulp_cut_overview
    cut_innercut = pad_or_pulp_cut_innercut
    cut_ic_pow = pad_or_pulp_cut_ic_pow
    cut_icgray_p = pad_or_pulp_cut_icgray_p
elif diffusion_model in kaliyuga_watercolor_model_names:
    cut_overview = watercolor_cut_overview
    cut_innercut = watercolor_cut_innercut
    cut_ic_pow = watercolor_cut_ic_pow
    cut_icgray_p = watercolor_cut_icgray_p


intermediate_saves = 0#@param{type: 'raw'}
intermediates_in_subfolder = True #@param{type: 'boolean'}
if type(intermediate_saves) is not list:
    if intermediate_saves:
        steps_per_checkpoint = math.floor((steps - skip_steps - 1) // (intermediate_saves+1))
        steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
        print(f'Will save every {steps_per_checkpoint} steps')
    else:
        steps_per_checkpoint = steps+10
else:
    steps_per_checkpoint = None

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0


width_height = width_height_for_256x256_models if diffusion_model in diffusion_models_256x256_list else width_height_for_512x512_models

#Get corrected sizes
side_x = (width_height[0]//64)*64;
side_y = (width_height[1]//64)*64;
if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

