import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

"""
    图像增强
    
    SSD的数据增强整体流程总体上包括 光学变换 与 几何变换 两个过程。
    1 光学变换包括亮度和对比度等随机调整，可以调整图像像素值的大小，并不会改变图像尺寸；
    2 几何变换包括扩展、裁剪和镜像等操作，主要负责进行尺度上的变化，
    3 最后再进行去均值操作。
    大部分操作都是随机的过程，尽可能保证数据的丰富性。
"""


# 用于计算box_a(一堆BBox即box_a的矩形框与同一个box_b的矩形框的交集面积)
def intersect(box_a, box_b):
    # 右下角坐标取坐标最小(向内缩)
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    # 左上角坐标取坐标最大(向内缩)
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    # 防止IoU等于0情况被计算错误(将不相交的负值裁剪为0)
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    # 返回一组交集的面积
    return inter[:, 0] * inter[:, 1]


# 用于计算一组IOU
# PS:计算IOU时无论是相对坐标还是绝对坐标不影响IOU大小(都等比例缩放不影响计算的IOU)
def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    # 计算的交集的面积
    inter = intersect(box_a, box_b)
    # 计算一组框的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *  # x_max-x_min
              (box_a[:, 3] - box_a[:, 1]))  # y_max-y_min
    # 计算这个框的面积
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    # 计算并集,加了两次就要减一次交集部分
    union = area_a + area_b - inter
    return inter / union  # [A,B]


"""
    Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
"""
"""
    将不同的增强方法组合在一起
    参数:
        transforms (List[Transform]): list of transforms to compose.
    例子:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
"""


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


"""
    在数据进行增强之前需要把图片的uchar类型转换为float类型
"""


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


'''
    去均值
    
    具体操作是减去每个通道的均值
'''


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


'''
    转为原图坐标
    把归一化后的box变回原图中的尺寸
'''


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


'''
    真实坐标转为归一化坐标
'''


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


'''
    几何变换3：固定缩放
    
    默认使用了300×300的输入大小。
    这里300×300的固定输入大小是经过精心设计的，
    可以恰好满足后续特征图的检测尺度，
    例如最后一层的特征图大小为1×1，负责检测的尺度则为0.9。
    原论文作者也给出了另外一个更精确的500×500输入的版本。
'''


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                   self.size))
        return image, boxes, labels


'''
    饱和度变化
    需要在HSV颜色空间下改变S的数字，同时S的变化倍数范围是[0.5,1.5]。
'''


class RandomSaturation(object):
    # 随机饱和度变化，需要输入图片格式为HSV
    def __init__(self, lower=0.5, upper=1.5):
        # lower表示增强倍数下限,upper表示增强倍数上限
        self.lower = lower
        self.upper = upper
        # 两次断言,防止出现非法传参
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        # 以50%的概率从0.5-1.5种选出n来增强饱和度n倍数
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


'''
    Hue变化需要在 HSV 空间下，通过加一个随机值，随机值区间为[-360,360]，改变H的数值，要保证H的取值范围是0-360。
'''


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


'''
    添加随机光照噪声
    具体做法是随机交换RGB三个通道的值
    图片更换通道，形成的颜色变化
'''


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            # 随机选取一个通道的交换顺序，交换图像三个通道的值
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


"""
    在进行亮度，对比度，色度调整之前，需要把色彩空间转换为HSV空间。
    当执行完亮度，对比度，色度调整之后，还需要把HSV颜色空间转回RGB颜色空间。
    HSV颜色空间：一种将RGB色彩空间中的点在倒圆锥体中的表示方法。把颜色分为三个参量，色相(Hue)、饱和度(Saturation)、亮度(Value)。
"""


class ConvertColor(object):
    # RGB和HSV颜色空间互转
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR->HSV
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)  # HSV->BGR
        else:
            raise NotImplementedError
        return image, boxes, labels


'''
    对比度变化
    图片的对比度变化，只需要在RGB空间下，乘上一个alpha值，alpha值区间[0.5,1.5)。
'''


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)  # 在 [x,y) 范围内, 随机生成一个实数，返回一个浮点数。
            image *= alpha
        return image, boxes, labels


'''
    亮度调整
    只需要在RGB空间下，加上一个delta值
    具体方法是以0.5的概率为图像中的每一个点加一个实数，该实数随机选取于[-32,32)区间中。
'''


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            # 随机选取一个位于[-32, 32)区间的数，相加到图像上
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


'''
    几何变换1：随机裁剪
    
    SSD的裁剪策略是从图像中随机裁剪出一块，
    1 需要保证该图像块至少与一个物体边框有重叠，
    重叠的比例从{0.1、0.3、0.7、0.9}中随机选取，
    2 同时至少有一个物体的中心点落在该图像块中。
    
    这种随机裁剪的好处在于，首先每一个图形块都要有物体，可以
    过滤掉不包含明显真实物体的图像；同时，不同的重叠比例也极大地
    丰富了训练集，尤其是针对物体遮挡的情况。

    图片随机裁剪在数据增强中有重要作用，算法的运行流程大致如下：

    1 随机选取裁剪框的大小；
    2 根据大小确定裁剪框的坐标；
    3 分析裁剪框和图片内部bounding box的交并比；
    4 筛选掉交并比不符合要求的裁剪框；
    5 裁剪图片，并重新更新bounding box的位置坐标；
'''


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # 使用原图 using entire original input image
            None,
            # 最小的IOU，和最大的IOU sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        # 原图的长宽
        height, width, _ = image.shape
        while True:
            # 随机选择一个切割模式 randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # 迭代50次 max trails (50)
            for _ in range(50):
                current_image = image
                # 随机一个宽度，一个高度
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # 判断长宽比在范围[0.5,2] aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # 切割的矩形大小，形状是[x1,y1,x2,y2] convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # 计算切割的矩形和 gt 框的iou大小 calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # 筛选掉不满足 overlap条件的 is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 从原图中裁剪矩形 cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # 所有GT的中心点坐标 keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # 以下代码原理是判断每个GT的中心坐标是否在裁剪框Rect里面，
                # 如果超出了，那么下面的mask就全为0，mask.any()返回false，即这次裁剪失败了。
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # 是否有合法的盒子
                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # 拷贝一份中心点落在裁剪区域中的GT
                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # 取出中心点落在裁剪区域中的GT对应的标签
                # take only matching gt labels
                current_labels = labels[mask]

                # 获取GT box和切割矩形的交点(左上角)  A点
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # 调节坐标系，让boxes的左上角坐标转变为切割后的坐标（切割图片以其左上角像素位原点产生新的坐标系）
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                # 获取GT box和切割矩形的交点(右下角)  B点
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # 调节坐标系，让boxes的左上角坐标变为切割后的坐标（切割图片以其左上角像素位原点产生新的坐标系）
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


'''
    几何变换2：图片随机尺度扩展
    
    扩展的具体过程：
    1 随机选择一个在[1,4)区间的数作为扩展比例，
    2 将原图像放在扩展后图像的右下角，
    3 其他区域填入每个通道的均值，即[104,117,123]
    4 将图像的bbox边框位置按照图像的平移进行同步平移
    
    设置一个大于原图尺寸的size，填充指定的像素值mean，
    然后把原图随机放入这个图片中，实现原图的扩充。
'''


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        # 利用random函数保证有一半的概率进行扩展
        if random.randint(2):
            return image, boxes, labels
        # 求取原图像在新图像中的左上角坐标值
        height, width, depth = image.shape
        # 随机选择一个位于[1,4)区间的比例值
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        # 新建一个图像，将均值与原图像素值依次赋予新图，
        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        # 填充均值
        expand_image[:, :, :] = self.mean
        # 填充原图
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image

        # 对应boxes边框也进行平移变换
        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


'''
    几何变换3：图像镜像
    
    将图片进行左右翻转，实现数据增强。
    
    图像镜像通常都是左右翻转。
'''


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            # 图片翻转 这里的::代表反向，即将第二个维度的数据width反向遍历，完成镜像
            # img [height, width, channels]
            image = image[:, ::-1]

            # boxes的坐标也需要相应改变 [xmin,ymin,xmax,ymax]->[width-xmax,ymin,width-xmin,ymax]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


'''
    针对图片的RGB空间，随机调换各通道的位置，实现不同灯光效果
'''


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


"""
    图片对比度，饱和度和色调变化的方式合并为一个类
    
    色相的随机调整与亮度很类似，都是随机地加一个数，而对比度与饱和度则是随机乘一个数。
    
    对色相与饱和度的调整是在HSV色域空间进行的。
    
    关于以上三者的调整顺序，SSD也给了一个随机处理，
    即有一半的概率对比度在另外两者之前，
    另一半概率则是对比度在另外两者之后。
"""


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # 对比度
            ConvertColor(transform='HSV'),  # BGR->HSV
            RandomSaturation(),  # 饱和度
            RandomHue(),  # 色相
            ConvertColor(current='HSV', transform='BGR'),  # HSV->BGR
            RandomContrast()  # 对比度
        ]
        self.rand_brightness = RandomBrightness()  # 亮度
        self.rand_light_noise = RandomLightingNoise()  # 随机光照噪声

    def __call__(self, image, boxes, labels):
        # 使用图像副本做数据增强操作
        im = image.copy()
        # 亮度扰动增强
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        # 如果随机到1(只可能随机到0,1)，就不执行pd的最后一个操作-对比度更改
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:  # 随机到0，就不执行pd的第一个操作-对比度更改
            distort = Compose(self.pd[1:])
        # 执行一系列pd中的操作
        im, boxes, labels = distort(im, boxes, labels)
        # 最后再执行一个图片更换通道，形成颜色变化
        return self.rand_light_noise(im, boxes, labels)


'''
    图像增强
'''


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # 首先将图像像素值从整型变成浮点型
            ConvertFromInts(),
            # 将标签中的边框从比例坐标变换为真实坐标
            ToAbsoluteCoords(),
            # 进行亮度、对比度、色相与饱和度的随机调整，然后随机调换通道
            PhotometricDistort(),
            Expand(self.mean),  # 随机扩展图像大小，图像仅靠右下方
            RandomSampleCrop(),  # 随机裁剪图像
            RandomMirror(),  # 随机左右镜像
            ToPercentCoords(),  # 从真实坐标变回比例坐标（归一化后坐标）
            Resize(self.size),  # 缩放到固定的300*300大小
            SubtractMeans(self.mean)  # 最后进行均值化
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
