# Reducing ANN-SNN Conversion Error through Residual Membrane Potential 

Codes for **Reducing ANN-SNN Conversion Error through Residual Membrane Potential** in *AAAI Conference on Artificial Intelligence (2023)*.

## Paper

**Abstract:** 

Spiking Neural Networks (SNNs) have received extensive academic attention due to the unique properties of low power consumption and high-speed computing on neuromorphic chips. Among various training methods of SNNs, ANN-SNN conversion has shown the equivalent level of performance as ANNs on large-scale datasets. However, unevenness error, which refers to the deviation caused by different temporal sequences of spike arrival on activation layers, has not been effectively resolved and seriously suffers the performance of SNNs under the condition of short time-steps. In this paper, we make a detailed analysis of unevenness error and divide it into four categories. We point out that the case of the ANN output being zero while the SNN output being larger than zero accounts for the largest percentage. Based on this, we theoretically prove the sufficient and necessary conditions of this case and propose an optimization strategy based on residual membrane potential to reduce unevenness error. The experimental results show that the proposed method achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet datasets. For example, we reach top-1 accuracy of 64.32% on ImageNet with 10-steps. To the best of our knowledge, this is the first time ANN-SNN conversion can simultaneously achieve high accuracy and ultra-low-latency on the complex dataset.

**Performance:**

|  Dataset  |   Arch    |   Para    |  ANN   |  T=1   |  T=2   |  T=4   |  T=8   |
| :-------: | :-------: | :-------: | :----: | :----: | :----: | :----: | :----: |
| CIFAR-100 |  VGG-16   | $\tau=4$  | 76.28% | 71.52% | 74.31% | 75.42% | 76.25% |
| CIFAR-100 | ResNet-20 | $\tau=4$  | 69.94% | 46.48% | 53.96% | 59.34% | 62.94% |
| ImageNet  |  VGG-16   | $\tau=14$ | 74.29% | 50.37% | 61.37% | 66.47% | 68.37% |
| ImageNet  | ResNet-34 | $\tau=8$  | 74.32% | 57.78% | 64.32% | 66.71% | 67.62% |



## Source Code

**Training an ANN model directly:**

```
python main.py --dataset {CIFAR10,CIFAR100,ImageNet} --datadir {YOUR DATASET DIR} --savedir {YOUR SAVED MODELS DIR} --net_arch {vgg16,resnet20,resnet34} --presim_len 4 --sim_len 32 --CUDA_VISIBLE_DEVICES {GPU ID} --batchsize 128 --direct_training
```

**Using SRP on a pretrained ANN model:**

```
python main.py --dataset {CIFAR10,CIFAR100,ImageNet} --datadir {YOUR DATASET DIR} --load_model_name {YOUR SAVED MODELS DIR/MODEL NAME} --net_arch {vgg16,resnet20,resnet34} --presim_len 4 --sim_len 32 --CUDA_VISIBLE_DEVICES {GPU ID}
```

