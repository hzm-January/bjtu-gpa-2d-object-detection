from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')


    # 生成所有的PriorBox，需要每一个特征图的信息
    def forward(self):
        mean = []
        # 遍历多尺度的 特征图: [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            # 遍历每个像素
            for i, j in product(range(f), repeat=2):
                # f_k为每个特征图的尺寸
                f_k = self.image_size / self.steps[k]
                # 求取每个box的中心坐标
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 对应{S_k, S_k}大小的PriorBox
                # aspect_ratio: 1 当 ratio==1的时候，会产生两个 box
                # r==1, size = s_k， 正方形
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 对应{√(S_k S_(k+1) ), √(S_k S_(k+1) )}大小的PriorBox
                # r==1, size = sqrt(s_k * s_(k+1)), 正方形
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 当 ratio != 1 的时候，产生的box为矩形
                # 剩余的比例为2、1/2、3、1/3的PriorBox
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # 转化为 torch的Tensor
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        # 归一化，把输出设置在 [0,1]
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
