FROM iflyopensource/dd:prep AS modelprep

FROM iflyopensource/disco_diffusion:aiges


RUN  pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/ 
# Install Python packages
RUN pip install imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 lpips datetime timm ipywidgets omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops wandb pandas ftfy opencv-python regex clip matplotlib

# Precache other big files
COPY --chown=root --from=modelprep /scratch/clip /root/.cache/clip
COPY --chown=root --from=modelprep /scratch/model-lpips/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# Copy over models used
COPY --from=modelprep /scratch/models /home/aiges/disco/models
COPY --from=modelprep /scratch/pretrained /home/aiges/disco/pretrained

ADD . /home/aiges/disco/


