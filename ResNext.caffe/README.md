

[从零开始搭建 ResNet 之 残差网络结构介绍和数据准备](http://www.cnblogs.com/Charles-Wan/p/6442294.html)


















### Notes

1. ResNet structure introduction

![Fig 1](http://okye062gb.bkt.clouddn.com/2017-08-15-060900.jpg)

上图显示了网络主体框架，可以看到一个残差模块（Fig 2）是由**两层卷积再加一个恒等映射组成的**。相同颜色块之间的 feature map 的大小是一样的，因此残差模块的输入输出的维度大小也是一样，可以直接进行相加（如 Fig 1中的实曲线）**网络延伸到不同颜色块时都要经过2倍下采样或者是 stride=2 的卷积，那么这时 feature map 的大小都会减半，但是卷积核的数量会增加一倍，这样是为了保持时间的复杂度，那么残差模块的输入和输出大小不一样的时应该要怎么办？这里采用论文中的 B 方法：用 1X1 的卷积核来映射到跟输出一样的维度（如 Fig 1中的虚曲线）**。

ResNet 的大体结构是还是参照 VGG 网络。


![Fig 2](http://okye062gb.bkt.clouddn.com/2017-08-15-061125.jpg)

本参考资料是搭建论文中 CIFAR10 实验的 ResNet，总共 20 层。结构如下：

![](http://okye062gb.bkt.clouddn.com/2017-08-15-061612.jpg)

每个 `Convx_x` 中都含有 3 个残差模块，每个模块的卷积核都是 3X3 大小的，pad 为 1，stride 为 1。`Con4_x` 的输出通过 `global_average_pooling` 映射到 64 个 1x1 大小的 feature map，最后再通过含有 10 个神经元的全连接层输出分类结果。


2. CIFAR 10 introduction

CIFAR10 数据库中的图片大小为 3 X 32 X 32（通道数 X 图像高度 X 图像宽度），训练数据为 50000 张，测试数据为 10000 张。此外还有 CIFAR100，那是分 100 类的图像数据库。

![](http://okye062gb.bkt.clouddn.com/2017-08-15-064010.jpg)

Create CIFAR10 lmdb database using official scripts.

```
0. Make sure you have cloned and built caffe repo on your home path
1. cd `CIFAR10` 
2. Run `./get_cifar10.sh`
3. Run `./create_cifar10.sh`
```

3. ResNet network build

<http://www.cnblogs.com/Charles-Wan/p/6535395.html>

























