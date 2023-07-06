import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


# --------------------------------------------#
#  图片格式变化:
#  读取时 :     jpg:  JpegImageFile(mode=RGB,np.shape(jpg)=(w,h,c=3))
#              png:  PngImageFile(mode=L,np.shape(png)=(w,h))
#  数据增强 :   jpg:  Image(mode=RGB,np.shape(jpg)=(w=512,h=512,c=3))
#              png:  Image(mode=L,np.shape(png)=(w=512,h=512))
#  后续处理 :   jpg:  np.array(np.shape(c=3,w=512,h=512))
#              png:  np.array(np.shape(w=512,h=512))
#              labels: np.array(np.shape(w=512,h=512,c=class_num)) one-hot
# --------------------------------------------#
class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeeplabDataset, self).__init__()
        # -------------------------------#
        #   Cityscapes数据集
        # -------------------------------#
        if  type(annotation_lines)==dict:
            self.annotation_type = annotation_lines['Type']
            self.annotation_image_lines = annotation_lines['image_lines']
            self.annotation_label_lines = annotation_lines['label_lines']
            self.length = len(self.annotation_image_lines)
            self.annotation_lines = []
        # -------------------------------#
        #   VOC数据集
        # -------------------------------#
        else:
            self.annotation_lines   = annotation_lines
            self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # -------------------------------#
        #   Cityscapes数据集
        # -------------------------------#
        if len(self.annotation_lines) == 0:
            annotation_image_line = self.annotation_image_lines[index]
            annotation_label_line = self.annotation_label_lines[index]
            # -------------------------------#
            #   格式 xxxx_xxxx_lines = ["xx_path_1/n","xx_path_2/n"...] -> ["xx_path_1","xx_path_2"...]
            # -------------------------------#
            image_name            = annotation_image_line.split()[0]
            label_name = annotation_label_line.split()[0]
            # -------------------------------#
            #   从文件中读取图像
            # -------------------------------#
            # 训练集
            if  self.annotation_type==0:
                jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "Cityspaces_leftImg8bit/train"), image_name))
                png = Image.open(os.path.join(os.path.join(self.dataset_path, "gtFine/train"), label_name))
            # 验证集
            elif self.annotation_type==1:
                jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "Cityspaces_leftImg8bit/val"), image_name))
                png = Image.open(os.path.join(os.path.join(self.dataset_path, "gtFine/val"), label_name))
        # -------------------------------#
        #    VOC数据集
        # -------------------------------#
        else:
            annotation_line = self.annotation_lines[index]
            name = annotation_line.split()[0]
            #-------------------------------#
            #   从文件中读取图像
            #-------------------------------#
            jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
            png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        #-------------------------------#
        #   数据增强 : 返回Image类型
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        # 将labels中不做分类(类与类交界的地方值为255)变为临界的值，方便后续转变为one-hot形式
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        #将labels转成one-hot形式: size = [w*h,num_classes + 1]
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        #reshape size = [w,h,num_classes + 1]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #转为RBG格式
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        # ------------------------------#
        #   对val数据集的操作
        # ------------------------------#
        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)
            # 图像放缩
            image       = image.resize((nw,nh), Image.BICUBIC)
            # 生成灰图:512*512
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            # 将为放缩后的图印到灰图
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            # 图像放缩 : 标签只能用Image.NEAREST差值
            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对val数据集的操作
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)

        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label



#---------------------------------#
# DataLoader中collate_fn使用:
# 先通过Dataset类里面的 __getitem__ 函数获取单个的数据，然后组合成batch。
# 再使用collate_fn所指定的函数对这个batch做一些操作
        # --------------------------------------------#
#  图片格式变化:
#  __getitem__ 从数据集中获得单个样本: batch =[img, png, labels]
#              jpg:  np.array(np.shape=(c=3,w=512,h=512))
#              png:  np.array(np.shape=(w=512,h=512))
#              labels: np.array(np.shape=(w=512,h=512,c=class_num)) one-hot
# collate_fn : 遍历所有样本,组成训练所用的batchs==[images[], pngs[], seg_labels[]]
#              images: torch.tensor(shape=(c=3,w=512,h=512))
#              pngs:   torch.tensor(shape=(w=512,h=512))
#              seg_labels: torch.tensor(shape=(w=512,h=512,c=class_num))
        # --------------------------------------------#
#---------------------------------#
def deeplab_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    #转换为tensor
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
