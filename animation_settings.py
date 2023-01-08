#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: animation_settings
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

### Animation Settings

animation_mode = 'None' #@param ['None', '2D', '3D', 'Video Input'] {type:'string'}

video_init_path = "init.mp4" #@param {type: 'string'}
extract_nth_frame = 2 #@param {type: 'number'}
persistent_frame_output_in_batch_folder = True #@param {type: 'boolean'}
video_init_seed_continuity = False #@param {type: 'boolean'}
#@markdown #####**Video Optical Flow Settings:**
video_init_flow_warp = True #@param {type: 'boolean'}
# Call optical flow from video frames and warp prev frame with flow
video_init_flow_blend =  0.999#@param {type: 'number'} #0 - take next frame, 1 - take prev warped frame
video_init_check_consistency = False #Insert param here when ready
video_init_blend_mode = "optical flow" #@param ['None', 'linear', 'optical flow']
# Call optical flow from video frames and warp prev frame with flow


interp_spline = 'Linear' #Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
angle = "0:(0)"#@param {type:"string"}
zoom = "0: (1), 10: (1.05)"#@param {type:"string"}
translation_x = "0: (0)"#@param {type:"string"}
translation_y = "0: (0)"#@param {type:"string"}
translation_z = "0: (10.0)"#@param {type:"string"}
rotation_3d_x = "0: (0)"#@param {type:"string"}
rotation_3d_y = "0: (0)"#@param {type:"string"}
rotation_3d_z = "0: (0)"#@param {type:"string"}
midas_depth_model = "dpt_large"#@param {type:"string"}
midas_weight = 0.3#@param {type:"number"}
near_plane = 200#@param {type:"number"}
far_plane = 10000#@param {type:"number"}
fov = 40#@param {type:"number"}
padding_mode = 'border'#@param {type:"string"}
sampling_mode = 'bicubic'#@param {type:"string"}

turbo_mode = False #@param {type:"boolean"}
turbo_steps = "3" #@param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 10 # frames

#insist turbo be used only w 3d anim.
if turbo_mode and animation_mode != '3D':
    print('=====')
    print('Turbo mode only available with 3D animations. Disabling Turbo.')
    print('=====')
    turbo_mode = False

#@markdown ####**2D Animation Settings:**
key_frames = True #@param {type:"boolean"}
max_frames = 10000#@param {type:"number"}
