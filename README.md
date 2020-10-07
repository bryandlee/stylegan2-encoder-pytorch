# stylegan2-encoder-pytorch


Implementation of **In-Domain GAN Inversion for Real Image Editing** based on **Seonghyeon Kim's Pytorch Implementation of StyleGAN2**

[[Paper](https://arxiv.org/pdf/2004.00049.pdf)] [[Official Code](https://github.com/genforce/idinvert)] [[StyleGAN2 Pytorch](https://github.com/rosinality/stylegan2-pytorch)]

## Train Encoder

```
python train_encoder.py
```

**0k iter**\
<img src="./imgs/0k.png" width="720">

**1M iter**\
<img src="./imgs/1M.png" width="720">\
[[checkpoint]](https://drive.google.com/file/d/1QQuZGtHgD24Dn5E21Z2Ik25EPng58MoU/view?usp=sharing)

## Interpolation

```
interpolate.ipynb
```

**Domain-Guided Encoder**\
<img src="./imgs/interpolation_domain_guided_encoder.png" width="480">

**In-Domain Inversion**\
<img src="./imgs/interpolation_idinversion_500steps.png" width="480">

**Inperpolation**\
<img src="./imgs/interpolation_results.png" width="720">



**Note:** The encoder architecture and loss weights are different from the original implemetation.
