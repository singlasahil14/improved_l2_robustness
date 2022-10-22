# Skew Orthogonal Convolutions

+ **Skew Orthogonal Convolution (SOC)** is a convolution layer that has an orthogonal Jacobian and achieves state-of-the-art standard and provably robust accuracy compared to other orthogonal convolutions. 
+ **Last Layer normalization (LLN)** leads to improved performance when the number of classes is large.
+ **Certificate Regularization (CR)** leads to significantly improved robustness certificates.
+ **Householder Activations (HH)** improve the performance for deeper networks.

## Prerequisites

+ Python 3.7 or 3.8
+ Pytorch 1.8 
+ einops. Can be installed using ```pip install einops```
+ NVIDIA Apex. Can be installed using ```conda install -c conda-forge nvidia-apex```
+ A recent NVIDIA GPU

## How to train 1-Lipschitz Convnets?

```python train_robust.py --conv-layer CONV --num-layers NUM_LAYERS --dataset DATASET --beta BETA last-layer LAST_LAYER```
+ CONV: bcop, cayley, soc
+ NUM_LAYERS: 5, 10, 15, 20, 25, 30, 35, 40
+ BETA: curvature regularization coefficient
+ LAST_LAYER: 'ortho', 'lln', 'crc_full'
+ Use ```crc_full``` to enable CRC-Lip
+ Use ```--fast-train``` to enable Fast SOC gradient computation
+ DATASET: cifar10/cifar100.

## Citations
If you find this repository useful for your research, please cite:

```
@inproceedings{
  improved2022,
  title={Improved techniques for deterministic l2 robustness},
  author={Sahil Singla and Soheil Feizi},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=ftKnhsDquqr}
}
```
