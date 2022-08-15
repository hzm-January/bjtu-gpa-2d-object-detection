# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import voc as cfg
# 这种用..来索引上级目录的方式值得使用
from ..box_utils import match, log_sum_exp


# 构造多任务损失函数用于训练模型
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    # num_classes：分类数
    # overlap_thresh：正负样本判断IOU阈值
    # prior_for_matching：
    # bkg_label：
    # neg_mining：
    # neg_pos：正样本是负样本的几倍
    # neg_overlap：
    # encode_target：
    # use_gpu：是否使用GPU
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap

        # import pdb
        # pdb.set_trace()
        # 回归参数编解码用的偏差参数
        self.variance = cfg['variance']

    # predictions：ssd.py中SSD类返回的一个元组
    # targets：经过数据增强并合并的GT框
    # targets的形状 [N,bbox_num,5]:其中bbox_num是这张图中的GT bbox数目，每张图不同
    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # 获取元组内的值
        # loc_data [N(Batch-Size),8732,4]
        # conf_data [N(Batch-Size),8732,21]
        # priors [8732,4]
        loc_data, conf_data, priors = predictions
        # num=batch-size
        num = loc_data.size(0)
        # 创建一个副本
        priors = priors[:loc_data.size(1), :]
        # num_priors=8732
        num_priors = (priors.size(0))  # 先验框个数
        # num_classes=21
        num_classes = self.num_classes  # 类别数

        # match priors (default boxes) and ground truth boxes
        # 1 首先匹配正负样本
        '''获取匹配每个prior box的 ground truth'''
        # 创建 loc_t 和 conf_t 保存真实box的位置和类别

        # loc_t [N(Batch-Size),8732,4]
        loc_t = torch.Tensor(num, num_priors, 4) # 将坐标转化为Tensor类型
        # 将类别转化为长Tensor类型便于分类损失的应用
        # conf_t [N(Batch-Size),8732,21]
        conf_t = torch.LongTensor(num, num_priors)
        # 遍历每一个batch中的每一张图
        for idx in range(num):
            # 取出这张图的的所有ground truth bbox坐标
            truths = targets[idx][:, :-1].data
            # 取出这张图的的所有ground truth bbox对应分类
            labels = targets[idx][:, -1].data
            # 取出四个生成的Priorbox坐标值
            defaults = priors.data
            # 得到每一个prior对应的truth,放到loc_t与conf_t中,conf_t中是类别,loc_t中是[matches, prior, variance]
            '''prior box 匹配 ground truth'''
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        # 如果选择使用GPU则将其送入GPU中训练
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets

        # 2 计算所有正样本的定位损失,负样本不需要定位损失

        '''计算正样本的数量'''
        # pos [N,8732]的bool值数组 若为True则是前景，False则为背景
        pos = conf_t > 0
        # 计算每一个batch中每一张图的8732个PriorBox中正样本的数目
        # num_pos [N,1]
        num_pos = pos.sum(dim=1, keepdim=True)

        # import pdb
        # pdb.set_trace()

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # 将pos_idx扩展为[32, 8732, 4],正样本的索引
        # pos [N,8732]--[N,8732,1]--[N,8732,4]
        # 复制操作，为了同时将为True(也就是参与位置回归的前景正样本）的四个坐标提取出来
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 预测的正样本box信息，提取相应下标的预测值的四个坐标并改变形状 loc_p [N*8732,4]
        loc_p = loc_data[pos_idx].view(-1, 4)
        # 真实的正样本box信息，提取相应下标的标签值的四个坐标并改变形状 loc_t [N*8732,4]
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 所有正样本的定位损失，使用Smooth L1损失函数计算回归损失
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # 3 对于类别损失,进行难样本挖掘,控制比例为1:3

        '''
            Target；
                下面进行hard negative mining
            过程:
                1、 针对所有batch的conf，按照置信度误差(预测背景的置信度越小，误差越大)进行降序排列;
                2、 负样本的label全是背景，那么利用log softmax 计算出logP，logP越大，则背景概率越低,误差越大;
                3、 选取误差较大大的top_k作为负样本，保证正负样本比例接近1:3;
        '''
        # Compute max conf across batch for hard negative mining
        # 所有prior的类别预测，将预测的分类坐标形状改成 batch_conf [N*8732,21]
        batch_conf = conf_data.view(-1, self.num_classes)
        # 计算类别损失.每一个的log(sum(exp(21个的预测)))-对应的真正预测值
        # 进入\layers\box_utils.py中查看log_sum_exp()函数
        # 筛选正负样本是用了log-softmax分类损失，但是为了防止数值溢出用了一次求和转化再减的操作
        # 后面的batch_conf.gather(1, conf_t.view(-1, 1))是将数字编码分类标签转化为独热码便签对应好分类损失函数
        # 这时候计算出来的分类损失是为了筛选正负样本
        # 使用logsoftmax，计算置信度,shape[b*M, 1]
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        # PS：源码中好像这两行写反了，我已经更改过，这样的话维度才匹配
        # 将loss_c转化为形状为 [N,8732]
        loss_c = loss_c.view(pos.size()[0], pos.size()[1])
        # 首先过滤掉正样本，设置正样本的分类损失为0
        # 把正样本排除，剩下的就全是负样本，可以进行抽样
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1) # shape[b, M]

        # 两次sort排序，能够得到每个元素在降序排列中的位置idx_rank
        # 获取负样本分类损失最大的那些框的索引
        _, loss_idx = loss_c.sort(1, descending=True)
        # idx_rank为排序后每个元素的排名
        # 再此排序loss_idx是为了便于排出rank，便于后续从大到小索引
        _, idx_rank = loss_idx.sort(1)

        # 抽取负样本
        # 统计正样本数目，每个batch中正样本的数目，shape[b,1]
        num_pos = pos.long().sum(1, keepdim=True)

        # 计算负样本数目并裁剪到[设定的正样本是负样本倍数*正样本数目,8731]之间
        # TODO 这个地方负样本的最大值不应该是pos.size(1)-num_pos?
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        # 选择每个batch中负样本的索引
        # pos [N,8732];neg [N,8732] 下面这是根据rank筛选负样本
        # 抽取前top_k个负样本，shape[b, M]
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 4 计算正负样本的类别损失

        # Confidence Loss Including Positive and Negative Examples
        # shape[b,M] --> shape[b,M,num_classes]
        # 都扩展为[32, 8732, 21]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 把预测提出来
        # 从预测值中索引正样本与相应的满足rank要求的负样本的类别置信度标签信息
        # conf_p [N*8732,21]
        # 提取出所有筛选好的正负样本(预测的和真实的)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # 对应的标签
        # 从标签值中索引正样本与相应的满足rank要求的负样本的类别数字标签信息
        # targets_weighted [N,8732]
        # PS:我认为这里也少了一个view操作，可能会报错
        # 下一行应该改为targets_weighted = conf_t[(pos+neg).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        '''计算conf交叉熵'''
        # 根据筛选出的正负样本计算真正的分类交叉熵损失
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # 计算最终的损失并返回
        N = num_pos.data.sum()
        if self.use_gpu:
            loss_l /= N.type('torch.cuda.FloatTensor')
            loss_c /= N.type('torch.cuda.FloatTensor')
        else:
            loss_l /= N.type('torch.FloatTensor')
            loss_c /= N.type('torch.FloatTensor')
        # 一个定位回归损失/一个分类损失
        return loss_l, loss_c
