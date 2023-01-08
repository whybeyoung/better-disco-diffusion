#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: setup3rd_mod
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
import sys


def setup_3rd_module(PROJECT_DIR):
    sys.path.append(f'{PROJECT_DIR}/CLIP')
    sys.path.append(f'{PROJECT_DIR}/open_clip/src')
    sys.path.append(f'{PROJECT_DIR}/guided-diffusion')
    sys.path.append(f'{PROJECT_DIR}/ResizeRight')
    sys.path.append(f'{PROJECT_DIR}/pytorch3d-lite')
    sys.path.append(f'{PROJECT_DIR}/MiDaS')
    sys.path.append(f'{PROJECT_DIR}/AdaBins')
