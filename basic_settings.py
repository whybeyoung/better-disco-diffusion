#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: basic_settings
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


## basic set

n_batches = 1 #@param{type: 'number'}
batch_name = 'TimeToDisco' #@param{type: 'string'}
steps = 20 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
width_height_for_512x512_models = [1280, 768] #@param{type: 'raw'}
clip_guidance_scale = 5000 #@param{type: 'number'}
tv_scale = 0#@param{type: 'number'}
range_scale = 150#@param{type: 'number'}
sat_scale = 0#@param{type: 'number'}
cutn_batches = 4#@param{type: 'number'}
skip_augs = False#@param{type: 'boolean'}

#@markdown ####**Image dimensions to be used for 256x256 models (e.g. pixelart models):**
width_height_for_256x256_models = [512, 448] #@param{type: 'raw'}

#@markdown ####**Video Init Basic Settings:**
video_init_steps = 100 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
video_init_clip_guidance_scale = 1000 #@param{type: 'number'}
video_init_tv_scale = 0.1#@param{type: 'number'}
video_init_range_scale = 150#@param{type: 'number'}
video_init_sat_scale = 300#@param{type: 'number'}
video_init_cutn_batches = 4#@param{type: 'number'}
video_init_skip_steps = 50 #@param{type: 'integer'}

init_image = None #@param{type: 'string'}
init_scale = 1000 #@param{type: 'integer'}
skip_steps = 10 #@param{type: 'integer'}
#@markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.*


animation_mode = 'None' #@param ['None', '2D', '3D', 'Video Input'] {type:'string'}


