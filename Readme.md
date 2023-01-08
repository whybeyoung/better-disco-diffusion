# better disco diffusion 

## background 

Thanks a lot for the work of disco diffusion, very cool...

However, the code is tough...

Although it may not have done much modification to the original code, I think it is easier to use, at least for AI engineers. The raw code may be elegant, but implementing a single function in hundreds of lines is daunting.

The current work is still not perfect, it needs you and me to iterate together, welcome your contribution

## target

make the origin dd's code more readable....

i split some implements, and using some class to wrapper the main logic...


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

## contact with me


* focus on:

[![ifly](https://avatars.githubusercontent.com/u/26786495?s=96&v=4)](https://github.com/iflytek)

* contact:

![weixin](https://raw.githubusercontent.com/berlinsaint/readme/main/weixin_ybyang.jpg)

  

