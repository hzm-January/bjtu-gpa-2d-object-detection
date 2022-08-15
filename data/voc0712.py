"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

# 判断python版本
if sys.version_info[0] == 2:  # python环境为python2
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# 类别
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# HOME为自动读取的储存项目目录的家目录，执行sh脚本后在data目录下生成VOCdevkit目录放数据集
VOC_ROOT = osp.join(HOME, "data/VOC/VOCdevkit/")


# 读取VOC中的xml标注
class VOCAnnotationTransform(object):
    # 继承自object类，可以把这个类实例化对象后接括号当作调用__call__函数来使用
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    """
        将VOC的annotation转换为bbox坐标张量，bbox坐标转化为归一化；
        将类别转化为用索引来表示的字典形式；
        参数列表：
        		class_to_ind: 类别的索引字典。
        		keep_difficult: 是否保留difficult=1的物体。
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        # 设置一个字典，一个类别名称标签对应着一个类别数字标签（0-19），共20个类别。
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        # 是否保持难处理的样本（也就是VOC数据集中标记为difficult的BBox要不要考虑）
        self.keep_difficult = keep_difficult

    """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
    """
    """
        参数列表：
                target: ET.Element对象，xml信息
                width: 图片宽度
                height: 图片高度
        返回值：
                该图片的 ground-truth bbox和labels信息，一个list，list中的每个元素是[bbox coords, class name id]
    """

    def __call__(self, target, width, height):
        # 参数target为一个ET.Element对象，可以用来迭代，也就是parse一个xml返回的对象
        # 参数width、height分别为这张图片的宽度与高度
        res = []  # 用于放置结果的列表，最终的shape为（一张图中GT的数目，4位置+1类别=5）
        # 遍历这张图xml文件中object下面的部分
        for obj in target.iter('object'):
            # 每一个obj均为一个BBox的信息
            # 判断该BBox是否为一个难处理样本并储存True/False到difficult中
            difficult = int(obj.find('difficult').text) == 1
            # 若不想要保留难处理样本且这个样本是难样本就跳过这个BBox
            if not self.keep_difficult and difficult:
                continue
            # 读取类别名字的字符串并转小写去除空格回车
            name = obj.find('name').text.lower().strip()
            # 读取BBox的四个坐标信息
            bbox = obj.find('bndbox')

            # 用于遍历获取左上角xy坐标与右下角xy坐标的列表
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            # 存储BBox坐标与类别信息的列表
            bndbox = []
            # 遍历pts
            for i, pt in enumerate(pts):  # bbox坐标归一化
                # 坐标要减一并转为整形
                cur_pt = int(bbox.find(pt).text) - 1
                # 将绝对坐标转化为相对总长度总宽度的相对坐标（便于在FeatureMap上对应）
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                # 将坐标添加到bndbox列表中
                bndbox.append(cur_pt)
            # 依据生成的类别字典与读取到的类别名称，将字符串类别标签转化为数字，便于投入分类模型
            label_idx = self.class_to_ind[name]
            # 加入类别标签
            bndbox.append(label_idx)
            # 将这个BBox的类别列表加入res列表中
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


# 数据集类，继承自Dataset基类,重写里面最重要的三个函数
class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    '''
        root                相对路径，用VOC_ROOT常量拼接目录寻找路径
        image_sets          数据集名称，默认 [('2007', 'trainval')]
        transform           图像增强算法，传一个类的实例化对象当函数用，这个实际传入的类在\ utils\ augmentations.py中
        target_transform    VOCAnnotationTransform类，用于读取并处理一张图片的标签
        dataset_name        需加载的数据集名称，默认VOC2007
    '''

    def __init__(self, root,
                 image_sets=[('2007', 'trainval')],  # 这里只使用2007数据集
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # 用%s来占位，一会从txt读取到文件名后填在这里
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')  # 图像标签读取路径
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')  # 图像读取路径
        self.ids = list()  # 储存图像id号的列表
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)  # 分别进入2007与2012数据集下
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):  # 正负样本
                self.ids.append((rootpath, line.strip()))  # (每个样本所在目录路径，样本无后缀名称)

    '''
        返回对应处理好的图像与标签
    '''

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    '''
        读取到了多少张图像则有多大的数据集搜索空间
    '''

    def __len__(self):
        return len(self.ids)

    '''
        1 根据图像id找到图片
        2 读取图像ground-truth信息并标准化
        3 图像增强
        4 并返回 图片信息、标注目标信息、高、宽
        返回值 (channels, height, width)，[xmin, ymin, xmax, ymax, label_ind], height, width
    '''

    def pull_item(self, index):  #
        # 找出index这张图片的(读取目录路径, 无后缀名称)
        img_id = self.ids[index]
        # 拼接该图像的xml文件读取路径并读取为ET.Element对象准备传入VOCAnnotationTransform类
        target = ET.parse(self._annopath % img_id).getroot()
        # 读取该图片，并存储为 numpy 格式
        img = cv2.imread(self._imgpath % img_id)
        # 获得图像的高度,宽度与通道数
        height, width, channels = img.shape
        # 对读取到的xml标签信息进行标准化转换，转换结果为[[bbox坐标信息xmin, ymin, xmax, ymax，类别标签],...]
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        # 对图像实施图像增强
        if self.transform is not None:
            # 先将target转化为numpy格式,便于传入增强模块处理
            target = np.array(target)
            # 在增强模块的类中进行处理(即utils\augmentations.py\SSDAugmentation类)
            # img [height, width, channels]
            # target [[bbox坐标信息xmin, ymin, xmax, ymax，类别标签],...]
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # 转换opencv读取的BGR图像为RGB图像
            img = img[:, :, (2, 1, 0)]  # to rgb
            # img = img.transpose(2, 0, 1)
            # 将四个坐标bbox与类别label合并在一起
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # 将img转为(C,H,W)的Tensor便于PyTorch训练
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    '''
        以PIL图像的方式返回下标为index的PIL格式原始图像
    '''

    def pull_image(self, index):  # 返回原始图片信息-PIL格式
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        # 依据图像的索引获取图像 [用于测试和预测(test.py中使用)]
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    '''
        Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
    '''
    '''
        返回索引为index的图像对应xml中的标注信息对象
        shape: [img_id, [(label, bbox coords),...]]
        例子: ('001718', [('dog', (96, 13, 438, 332))])
    '''

    def pull_anno(self, index):  # 返回原始图片标签
        # 依据图像的索引获取标签值与图片名 [用于测试和预测(test.py中使用)]
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        # 第二个参数是没有用的占位参数，第三个表示保留difficult样本，返回GT信息
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    '''
        Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
    '''
    '''
        以Tensor的形式返回索引为index的原始图像，调用unsqueeze_函数
    '''

    def pull_tensor(self, index):  # 返回原始图片的tensor
        # 将图像转化为Tensor并加一个维度用于预测
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
