# better disco diffusion 

## background 

Thanks a lot for the work of disco diffusion, very cool...

However, the code is tough...

Although it may not have done much modification to the original code, I think it is easier to use, at least for AI engineers. The raw code may be elegant, but implementing a single function in hundreds of lines is daunting.

The current work is still not perfect, it needs you and me to iterate together, welcome your contribution

## target

make the origin dd's code more readable....

i split some implements, and using some class to wrapper the main logic...

## todolist

- [x] split all in one file , and make the main logic ok..
- [ ] simplify  the `do_run` function.


## pull from docker hub
the image is so big which have the whole models that you don't need to download..

```
docker pull iflyopensource/disco_diffusion:v2
cd /home/aiges/disco
python run.py
```


## or you can build  docker image

```
git clone git@github.com:whybeyoung/better-disco-diffusion.git

git submodule update --init --recursive

docker build . -t dd:latest
```

## notebook

[Get started](docs/get_started.ipynb)
![img.png](docs/img.png)
![img2.png](docs/img2.png)
![img3.png](docs/img3.png)


## Other
text_prompts: 是文生图的输入参数，该参数具有一定的结构设计,它描述图像应该是什么样子的短语、句子或单词和短语字符串。人工智能将分析这些单词，并将引导扩散过程朝着靠近描述的方向上优化。

    E.g. "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."

文本提示词大致遵循这样一个结构：[主题]、[介词细节]、[设置]、[基本修饰和艺术家]；这是您尝试的一个很好的起点。


- 主题[subject]：画面的主体对象如一座城堡

- 介词细节[prepositional details]：主体所处环境如花海当中的一座城堡

- 设置[setting]：主体对象的活动或状态如花海当中举办盛大庆典/逐渐荒凉凋敝的一座城堡以及其具体修饰如花海当中逐渐荒凉凋敝的一座巨大的/英式城堡

- 基本修饰[meta modifiers]：提示DD该图像的艺术风格、或传达的情感如一种伤感的风格、胶卷相机拍照的风格、抽象风格

- 艺术家/创作者[artist]：提示DD该图像类似于某个具体创作者的风格如宫崎骏、梵高、莫奈

）


## contact with me


* focus on:

[![ifly](https://avatars.githubusercontent.com/u/26786495?s=96&v=4)](https://github.com/iflytek)

* contact:

![weixin](https://raw.githubusercontent.com/berlinsaint/readme/main/weixin_ybyang.jpg)

## link

- https://blog.csdn.net/qq_42635142/article/details/125481410
- https://www.bilibili.com/read/cv16730942?from=search
- http://www.gaoxigang.com/?p=435