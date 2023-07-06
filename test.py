# import torch
# import torch.nn.functional as F

# x=torch.randint(0,5,(2,3,4,4))*1.0
# print(x)
# x = torch.mean(x, 2, True) # dim=2 : 沿列取平均，即转为一行
# print(x)
# x = torch.mean(x, 3, True) # dim=3 : 沿行取平均，即转为一列
# print(x)
# x = F.interpolate(x, (4, 4), None, 'bilinear', True)
# print(x)

# from PIL import Image
# while True:
#     img = input('Input image filename:')
#     # img='VOCdevkit/VOCdevkit/VOC2007/JPEGImages/000033.jpg'
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         image.show()


# import cv2
# cap = cv2.VideoCapture(0)
# while(cap.isOpened()):
#     retval, frame = cap.read()
#     cv2.imshow('Live', frame)
#     if cv2.waitKey(5) >= 0:
#         break

# import colorsys
# hsv_tuples = [(x / 5, 1., 1.) for x in range(5)]
# print(hsv_tuples)
# colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
# print(colors)
# colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
# print(colors)

# from PIL import Image
# image_path  = "img/street.jpg"
# image = Image.open(image_path)
# image.show()
#
# iw, ih = image.size
# print(iw, ih)
# w, h = [512,512]
#
# scale = min(w / iw, h / ih)
# nw = int(iw * scale)
# nh = int(ih * scale)
#
# image = image.resize((nw, nh), Image.BICUBIC)
# image.show()
# new_image = Image.new('RGB', [512,512], (128, 128, 128))
# image.show()
# new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
# new_image.show()

# import numpy as np
# a = np.arange(24).reshape(2,3,4)
# print(a)
# print()
# pr = a.argmax(axis=0)
# print(pr)
# print()
# pr = a.argmax(axis=1)
# print(pr)
# print()
# pr = a.argmax(axis=2)
# print(pr)
# print()
# pr = a.argmax(axis=-1)
# print(pr)

# import cv2
# import matplotlib.pyplot as plt
#
# # img = cv2.imread('VOCdevkit/VOC2007/SegmentationClass/2010_004469.png',cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('VOCdevkit/gtFine/test/berlin/berlin_000000_000313_gtFine_labelIds.png',cv2.IMREAD_GRAYSCALE)
# # img = cv2.imread('VOCdevkit/gtFine/test/berlin/berlin_000000_000019_gtFine_labelTrainIds.png',cv2.IMREAD_GRAYSCALE)
# plt.imshow(img)
# plt.show()
# print(img.shape)

# new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# plt.imshow(new_img)
# new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# gray = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
# plt.imshow(img)
# plt.imshow(new_img)
# plt.imshow(gray)

# a=torch.Tensor([[[1,2,3],[4,5,6]]])
# print(a.shape)
# a=a.view(3,2)
# print(a)
# a=a.view(2,-1)
# print(a)


# loss = torch.nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# print(input,'\n',target,'\n',output)
# output.backward()

# import torch
# import torch.nn as nn
# import math
# loss = nn.CrossEntropyLoss()
# input = torch.randn(1, 3, requires_grad=True)
# print(input)
# input = torch.softmax(input,-1)
# print(input)
# target = torch.empty(1, dtype=torch.long).random_(3)
# output = loss(input, target)
#
#
# print("要计算loss的类别:")
# print(target)
# print("计算loss的结果:")
# print(output)
# print("手算loss的结果:")
# floss=-input[0][target]+math.log(math.exp(input[0][0])+math.exp(input[0][1])+math.exp(input[0][2]))
# print(floss)
# ffloss=-math.log(input[0][target])
# print(ffloss)

# import torch
# A = torch.arange(10*4,dtype=torch.float32).reshape(2, 5, 4)
# print(A.shape, A.sum())
# print(A)
# B = torch.arange(10*1,dtype=torch.float32).reshape(2, 5, 1)
# print(B.shape, B.sum())
# print(B)
#
# temp= A[..., :-2]
# print(temp)
# temp= A[..., :-1] * B
# print(temp)
# B = torch.arange(10*1,dtype=torch.float32).reshape(2, 5, 1)
# tp = torch.sum(B, axis=[0, 2],keepdim=True)
# print(tp)

# import matplotlib.pyplot as plt
# import os  # 注意要输入OS模块
# figure_save_path = "file_fig"
# if not os.path.exists(figure_save_path):
#     os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
# plt.savefig(os.path.join(figure_save_path, 'exam.png'))  # 第一个是指存储路径，第二个是图片名字
# from PIL import Image
# im = Image.open("D:\Python_Deeplabv3p\img\street.jpg")
# #此时返回一个新的image对象，转换图片模式
# image=im.convert('RGB')
# # image.show()
# #调用save()保存
# image.save('D:\Python_Deeplabv3p\img\street2.rgb')


# import cv2
# image = cv2.imread("VOCdevkit/VOCdevkit/VOC2007/JPEGImages/000033.jpg")     #"432.bmp"没有指定路径，是因为该图片的路径和代码的路径一致 数字0，如上图所解释。
# cv2.imshow("test pic",image)         #其中“试错代码”为显示的窗口名字
# cv2.waitKey()                       #延长窗口显示时间，以毫秒为单位，0为无限延长


# img = input('Input image filename:')
# print(img)
# img=img[:-4]+"_pre"+img[-4:]
# print(img)
#
# torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# torch.device('cuda:30' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(2)


# import torch
#
# pthfile = 'model_data/deeplab_mobilenetv2.pth'  # .pth文件的路径
# model = torch.load(pthfile, torch.device('cuda'))
# print('type:')
# print(type(model))  # 查看模型字典长度
# print('length:')
# print(len(model))
# print('key:')
# for k in model.keys():  # 查看模型字典里面的key
#     print(k)
#
# # print('value:')
# # for k in model:  # 查看模型字典里面的value
# #     print(k, model[k])

# dic_1 ={"A":1,"B":2,"C":3}
# dic_2 ={"A":3,"B":4,"C":5,"D":6,"E":7}
# print(dic_1,dic_2)
#
# dic_3 = {k: v for k, v in dic_2.items() if k in dic_1 and dic_1[k]!=v}
# print(dic_1,dic_2,dic_3)

# print("\n\033[1;31;47m   Check  \033[0m")
# print("\n\033[4;32;46m   Check  \033[0m")
# print("\n\033[5;33;40m   Check  \033[0m")

# from tqdm import tqdm
# import time
# epoch_step=100
# epoch=1
# Epoch=50
# pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
# pbar.set_postfix(**{"para_a":1,"para_b":2})
# for i in range(1,epoch_step):
#     pbar.set_postfix(**{"para_a": 1, "para_b": 2,f'test {i}/{epoch_step}':i})
#     pbar.update(1)
#     time.sleep(1)


# import cv2
# import numpy as np
# import torch
# from PIL import Image
#
# def preprocess_input(image):
#     image /= 255.0
#     return image
#
# num_classes=21
# jpg = Image.open("VOCdevkit/VOC2007/JPEGImages/2007_000032.jpg")
# png = Image.open("VOCdevkit/VOC2007/SegmentationClass/2007_000032.png")
#
# j_s=np.shape(jpg)
# p_s=np.shape(png)
#
# label   = Image.fromarray(np.array(png))
#
# jpg =preprocess_input(np.array(jpg, np.float64))
# jpg = np.transpose(jpg, [2, 0, 1])
# png = np.array(png)
# png[png >= num_classes] = num_classes
#
# w,h= png.shape
# seg_labels = np.eye(num_classes + 1)[png.reshape([-1])]
# seg_labels = seg_labels.reshape((int(w), int(h), num_classes + 1))
#
# print("TT")

# ------------------------------------------------------#
#  测试 Cityscapes 数据集
# ------------------------------------------------------#
# import cv2
# import numpy as np
# import torch
# from PIL import Image
#
# num_classes=21
# img = Image.open("VOCdevkit/Cityspaces_leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png")
# png_color = Image.open("VOCdevkit/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png")
# png_labelIds = Image.open("VOCdevkit/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png")
# png_instanceIds = Image.open("VOCdevkit/gtFine/train/aachen/aachen_000000_000019_gtFine_instanceIds.png")
#
# # img.show()
# # png_color.show()
# # png_labelIds.show()
# # png_instanceIds.show()
#
# img=np.array(img)
# png_color=np.array(png_color)
# png_labelIds=np.array(png_labelIds)
# png_instanceIds=np.array(png_instanceIds)
#
# print()

# j_s=np.shape(jpg)
# p_s=np.shape(png)
#
# label   = Image.fromarray(np.array(png))
#
# jpg =preprocess_input(np.array(jpg, np.float64))
# jpg = np.transpose(jpg, [2, 0, 1])
# png = np.array(png)
# png[png >= num_classes] = num_classes
#
# w,h= png.shape
# seg_labels = np.eye(num_classes + 1)[png.reshape([-1])]
# seg_labels = seg_labels.reshape((int(w), int(h), num_classes + 1))
#
# print("TT")



# ------------------------------------------------------#
#  测试分割样本名字的函数
# ------------------------------------------------------#
# from utils.utils import ReadTruepath_from_txt
# ReadTruepath_from_txt()


# ------------------------------------------------------#
#  分割数据集样本名字
# ------------------------------------------------------#
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# train_dirs = ["aachen/", "bochum/"]
# val_dirs = ["frankfurt/", "lindau/"]
#
# image_train_path = "VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/train/"
# image_val_path = "VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/val/"
# label_train_path = "VOCdevkit/Cityscapes/gtFine/train/"
# label_val_path = "VOCdevkit/Cityscapes/gtFine/val/"
#
#
# image_train_dir ="VOCdevkit/Cityscapes/image_train.txt"
# image_val_dir ="VOCdevkit/Cityscapes/image_val.txt"
# label_train_dir ="VOCdevkit/Cityscapes/label_train.txt"
# label_val_dir ="VOCdevkit/Cityscapes/label_val.txt"
# #
#
# # # 获得训练集的原图名字
# # with open(image_train_dir, 'w') as f:
# #     for son_dir in train_dirs:
# #         true_dir = image_train_path + son_dir
# #         file_names = os.listdir(true_dir)
# #         for file_name in file_names:
# #             file_name = str(son_dir)+file_name
# #             f.write(file_name+'\n')
# #     f.close()
# #
# # # 获得验证集的原图名字
# # with open(image_val_dir, 'w') as f:
# #     for son_dir in val_dirs:
# #         true_dir = image_val_path + son_dir
# #         file_names = os.listdir(true_dir)
# #         for file_name in file_names:
# #             file_name = str(son_dir)+file_name
# #             f.write(file_name + '\n')
# #     f.close()
# #
# # # 获得训练集的标签名字
# # with open(label_train_dir, 'w') as f:
# #     for son_dir in train_dirs:
# #         true_dir = label_train_path + son_dir
# #         file_names = os.listdir(true_dir)
# #         for file_name in file_names:
# #             if  file_name[-19:]=="gtFine_labelIds.png":
# #                 file_name = str(son_dir) + file_name
# #                 f.write(file_name + '\n')
# #     f.close()
# #
# # # 获得验证集的标签名字
# # with open(label_val_dir, 'w') as f:
# #     for son_dir in val_dirs:
# #         true_dir = label_val_path + son_dir
# #         file_names = os.listdir(true_dir)
# #         for file_name in file_names:
# #             if file_name[-19:] == "gtFine_labelIds.png":
# #                 file_name = str(son_dir) + file_name
# #                 f.write(file_name + '\n')
# #     f.close()

# ------------------------------------------------------#
#  plt 同时显示多幅图像
# ------------------------------------------------------#
# with open(image_train_dir, 'r') as f:
#     image_train_lines=f.readlines()
# with open(image_val_dir, 'r') as f:
#     image_val_lines=f.readlines()
# with open(label_train_dir, 'r') as f:
#     label_train_lines=f.readlines()
# with open(label_val_dir, 'r') as f:
#     label_val_lines=f.readlines()
#
# image_train = Image.open(image_train_path+image_train_lines[0].split()[0])
# image_val = Image.open(image_val_path+image_val_lines[0].split()[0])
# label_train = Image.open(label_train_path+label_train_lines[0].split()[0])
# label_val = Image.open(label_val_path+label_val_lines[0].split()[0])
#
# plt.subplot(2,2,1)
# plt.imshow(image_train)
# plt.subplot(2,2,2)
# plt.imshow(image_val)
# plt.subplot(2,2,3)
# plt.imshow(label_train)
# plt.subplot(2,2,4)
# plt.imshow(label_val)
# plt.show()


# ------------------------------------------------------#
#  测试 分割数据集样本名字
# ------------------------------------------------------#
# with open(image_train_dir, 'r') as f:
#     train_lines = f.readlines()
#
# print(len(train_lines))
# for i in range(0,10):
#     name = train_lines[i].split()[0]
#     print('No. %d :'%(i),name)

# ------------------------------------------------------#
#  测试 判断字典和列表 类型
# ------------------------------------------------------#
# l=[1,2,3]
# d={'a':1,'b':2}
# print(type(l),type(d))
# print(type(l)==list)
# print(type(l)==dict)
# print(type(d)==list)
# print(type(d)==dict)


# ------------------------------------------------------#
#  训练中断后,加载log中参数测试
# ------------------------------------------------------#
import torch

# pthfile = 'model_data/deeplab_mobilenetv2.pth'  # .pth文件的路径
# model = torch.load(pthfile, torch.device('cuda'))
# print(type(model))  # 查看模型字典长度
# print('length:')
# print(len(model))
# print('key:')
# for k in model.keys():  # 查看模型字典里面的key
#     print(k)
#
# pthfile = 'logs/last_epoch_weights.pth'  # .pth文件的路径
# model2 = torch.load(pthfile, torch.device('cuda'))
# print(type(model2))  # 查看模型字典长度
# print(len(model2))
# print('key:')
# for k in model2.keys():  # 查看模型字典里面的key
#     print(k)

# ------------------------------------------------------#
#  测试ptimizer.param_groups
# ------------------------------------------------------#
import torch.optim as optim
from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
# model = DeepLab(num_classes=21, backbone='mobilenet', downsample_factor=16,
#                 pretrained=False)
# optimizer=optim.SGD(model.parameters(), 1e-2, momentum=0.999, nesterov=True,
#                              weight_decay= 1e-4)
# lr_scheduler_func = get_lr_scheduler("cos", 1e-2, 1e-3, 100)
# lr = lr_scheduler_func(5)
#
# print(type(optimizer.param_groups))
# print(optimizer.param_groups)
# print(len(optimizer.param_groups))
# print(optimizer.param_groups[0]['lr'])
# for param_group in optimizer.param_groups:
#     param_group['lr'] = lr

# ------------------------------------------------------#
#  测试SummaryWriter
# ------------------------------------------------------#
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('logs/loss_2023_03_05_15_21_13') #建立一个保存数据用的东西，save是输出的文件名
# dummy_input = torch.rand(2, 3, 512, 512)  # 网络中输入的数据维度
# model = DeepLab(num_classes=21, backbone='mobilenet', downsample_factor=16,
#                 pretrained=False)
# with SummaryWriter(comment='LeNet') as w:
#     w.add_graph(model, (dummy_input,))  # net是你的网络名

# ------------------------------------------------------#
#  测试plt
# ------------------------------------------------------#
# from matplotlib import pyplot as plt
# import scipy.signal
# import  random
# print(random.random())
# losses =[100-a**2+random.randint(30,40)for a in range(0,2)]
# val_loss = [100-a**2+10+random.randint(25,35) for a in range(0,2)]
# iters = range(len(losses))
# plt.figure()
# plt.plot(iters, losses, 'red', linewidth=2, label='train loss')
# plt.plot(iters, val_loss, 'coral', linewidth=2, label='val loss')
# # plt.show()
# try:
#     if len(losses) < 25:
#         num = 5
#     else:
#         num = 15
#
#     plt.plot(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle='--', linewidth=2,
#              label='smooth train loss')
#     plt.plot(iters, scipy.signal.savgol_filter(val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
#              label='smooth val loss')
# except Exception as e:
#     print(e)
#     pass
#
# plt.grid(True)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc="upper right")
#
# plt.show()
# plt.cla()
# plt.close("all")

# ------------------------------------------------------#
#  测试os.path.join 带‘/’
# ------------------------------------------------------#
# import os
# from PIL import Image
#
# # 路径 = 'VOCdevkit/VOC2007/SegmentationClass'
# gt_dir = os.path.join('VOCdevkit', "VOC2007/SegmentationClass")
# gt_dir_2 = os.path.join('VOCdevkit', "VOC2007/SegmentationClass/")
# print(gt_dir)
# print(gt_dir_2)
# x= '2007_000032'
# pic_1 = Image.open(os.path.join(gt_dir, x + ".png"))
# pic_2 = Image.open(os.path.join(gt_dir, x + ".png"))
#
# plt.subplot(1,2,1)
# plt.imshow(pic_1)
# plt.subplot(1,2,2)
# plt.imshow(pic_2)
# plt.show()

# ------------------------------------------------------#
#  测试np.sum
# ------------------------------------------------------#
# import numpy as np
# l =np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(l.sum(1))
# print(l.sum(0))


# ------------------------------------------------------#
#  测试预测结果与原图叠加
# ------------------------------------------------------#
# from PIL import Image
# import matplotlib
# import  numpy as np
# matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
# import matplotlib.pyplot as plt
# import cv2
# import os
#
# path='VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
# path2='VOCdevkit/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'
# path3='VOCdevkit/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png'
# image=Image.open(path)
# array=np.array(image)
# image4=Image.open(path3)
# array4=np.array(image4)
#
# image2=Image.open(path2)
# array2=np.array(image2)
# array2[array2 >=21] = 21
#
#
# colors = [ (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30),
#         (220,220,  0), (107,142, 35), (152,251,152),( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
# 		(  0, 60,100), (  0,  0, 90), (  0,  0,110),(  0, 80,100), (  0,  0,230), (119, 11, 32), (128 , 128 , 128)]
# color_np = np.array(colors, np.uint8)
# ss = np.reshape(array2, [-1])
# seg_img = np.reshape(color_np[ss], [1024, 2048, -1])
# seg_img   = Image.fromarray(np.uint8(seg_img))
#
# image3 = Image.blend(image, seg_img, 0.5)
# array3=np.array(image3)
# plt.figure()
# plt.subplot(1,4,1)
# plt.imshow(array)
# plt.subplot(1,4,2)
# plt.imshow(array2)
# plt.subplot(1,4,3)
# plt.imshow(array4)
# plt.subplot(1,4,4)
# plt.imshow(array3)
# plt.show()

# ------------------------------------------------------#
#  测试miou预测的结果
# ------------------------------------------------------#
#from PIL import Image
# import matplotlib
# import  numpy as np
# matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
# import matplotlib.pyplot as plt
# import cv2
# import os
#
# path='VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
# path2='VOCdevkit/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'
# path3='VOCdevkit/Cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png'
# image=Image.open(path)
# array=np.array(image)
# image2=Image.open(path2)
# array2=np.array(image2)
#
# cvimage=cv2.imread(path)
# cvimage2=cv2.imread(path2)
# temp_cv=cvimage2.transpose(2, 0, 1)
# plt.figure()
# plt.subplot(1,4,1)
# plt.imshow(array)
# plt.subplot(1,4,2)
# plt.imshow(array2)
# plt.subplot(1,4,3)
# plt.imshow(cvimage)
# plt.subplot(1,4,4)
# plt.imshow(cvimage2)
# plt.show()



# path3='VOCdevkit/VOC2007/SegmentationClass/2007_000033.png'
#
# # path='VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
# # path2='VOCdevkit/VOC2007/JPEGImages/2007_000033.jpg'
# # path3='VOCdevkit/VOC2007/SegmentationClass/2007_000033.png'
#
#
# path='miou_out_cityscapes/detection-results/frankfurt_000000_000294_leftImg8bit.png'
# image=Image.open(path)
# rgb_image=image.convert('RGB')
# image.save(os.path.join('.temp_miou_out', 'tt.jpg'))

# img = cv2.imread(path)
# cv2.imwrite("./" + newpath + "/" + portion[0] + '.' + picture_type, img)

# image=Image.open(path)
# png_array=np.array(image)
# rgb_image=image.convert('RGB')
# rgb_array=np.array(rgb_image)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(image)
# plt.subplot(1,2,2)
# plt.imshow(rgb_image)
# plt.show()
#
# image=Image.open(path)
# image2=Image.open(path2)
# image3=Image.open(path3)
# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(image)
# plt.subplot(1,3,2)
# plt.imshow(image2)
# plt.subplot(1,3,3)
# plt.imshow(image3)
# plt.show()
#
# image=np.array(image)
# image2=np.array(image2)
# image3=np.array(image3)
# print('')

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:39:03 2019

@author: wsb
"""

# import cv2
# import os

# print('----------------------------------------------------')
# print('程序的功能为：将该目录下输入的文件内的图片转为指定格式')  # 目前我测试了jpg转化为png和png转化为jpg。
# print('转化结果保存在当前目录下的new_picture内')
# print('----------------------------------------------------')
#
# son = raw_input('请输入需要转化的文件夹名：')
# picture_type = raw_input('请输入想要将图片转化的类型：')
# daddir = './'
# path = daddir + son
#
# newpath = "new_picture"
# if not os.path.exists(newpath):
#     os.mkdir(newpath)
#
# path_list = os.listdir(path)
# number = 0  # 统计图片数量
# for filename in path_list:
#     number += 1
#     portion = os.path.splitext(filename)
#     print('convert  ' + filename + '  to ' + portion[0] + '.' + picture_type)
#     img = cv2.imread(path + "/" + filename)
#     cv2.imwrite("./" + newpath + "/" + portion[0] + '.' + picture_type, img)
# print("共转化了%d张图片" % number)
# print('转换完毕，文件存入 ' + newpath + ' 中')
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ------------------------------------------------------#
#  测试EvalCallback
# ------------------------------------------------------#
# import os
# import datetime
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader
#
# from nets.deeplabv3_plus import DeepLab
# from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
#                                      weights_init)
# from utils.callbacks import LossHistory, EvalCallback
# from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
# from utils.utils import download_weights, show_config
# from utils.utils_fit import fit_one_epoch
#
# image_train_dir = "VOCdevkit/Cityscapes/image_train.txt"
# image_val_dir = "VOCdevkit/Cityscapes/image_val.txt"
# label_train_dir = "VOCdevkit/Cityscapes/label_train.txt"
# label_val_dir = "VOCdevkit/Cityscapes/label_val.txt"
# with open(image_train_dir, 'r') as f:
#     image_train_lines = f.readlines()
# with open(image_val_dir, 'r') as f:
#     image_val_lines = f.readlines()
# with open(label_train_dir, 'r') as f:
#     label_train_lines = f.readlines()
# with open(label_val_dir, 'r') as f:
#     label_val_lines = f.readlines()
#
# # 格式 xxxx_xxxx_lines = ["xx_path_1/n","xx_path_2/n"...]
# assert len(image_train_lines) == len(label_train_lines) and len(image_val_lines) == len(label_val_lines)
#
# num_train = len(image_train_lines)
# num_val = len(image_val_lines)
#
# train_lines = {'Type': 0, 'image_lines': image_train_lines, 'label_lines': label_train_lines}
# val_lines = {'Type': 1, 'image_lines': image_val_lines, 'label_lines': label_val_lines}
#
# num_classes=21
# backbone='mobilenet'
# downsample_factor=16
# pretrained=False
# input_shape=[512,512]
# VOCdevkit_path  = 'VOCdevkit/Cityscapes'
# log_dir = 'logs/City_loss_2023_03_06_21_37_09'
# Cuda = True
# eval_flag = True
# eval_period =1
#
# model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
#                 pretrained=pretrained)
#
# model_path='logs/best_epoch_weights.pth'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # 第一步：读取当前模型参数
# model_dict = model.state_dict()
# # 第二步：读取预训练模型
# pretrained_dict = torch.load(model_path, map_location=device)
# load_key, no_load_key, temp_dict = [], [], {}
# for k, v in pretrained_dict.items():
#     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#         temp_dict[k] = v
#         load_key.append(k)
#     else:
#         no_load_key.append(k)
# # 第三步：使用预训练的模型更新当前模型参数
# model_dict.update(temp_dict)
# # 第四步：加载模型参数
# model.load_state_dict(model_dict)
# print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
# print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
# print("\n\033[1;33;40m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
# model_train = torch.nn.DataParallel(model)
# cudnn.benchmark = True
# model_train = model_train.cuda()
#
# eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
#                                          eval_flag=eval_flag, period=eval_period)
# model_train.eval()
# epoch = 1
# eval_callback.on_epoch_end(epoch + 1, model)



# ------------------------------------------------------#
#  测试 plt 柱状图
# ------------------------------------------------------#
# from PIL import Image
# # import matplotlib
# # matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
# # def adjust_axes(r, t, fig, axes):
# #     bb                  = t.get_window_extent(renderer=r)
# #     text_width_inches   = bb.width / fig.dpi
# #     current_fig_width   = fig.get_figwidth()
# #     new_fig_width       = current_fig_width + text_width_inches
# #     propotion           = new_fig_width / current_fig_width
# #     x_lim               = axes.get_xlim()
# #     axes.set_xlim([x_lim[0], x_lim[1] * propotion])
# #
# #
# # def draw_plot_func(values, name_classes, plot_title, x_label, tick_font_size = 12, plt_show = True):
# #     #获取当前图表对象
# #     fig     = plt.gcf()
# #     axes    = plt.gca()
# #     #横向的柱状图:
# #     plt.barh(range(len(values)), values, color='royalblue')
# #     plt.title(plot_title, fontsize=tick_font_size + 2)
# #     plt.xlabel(x_label, fontsize=tick_font_size)
# #     plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
# #     r = fig.canvas.get_renderer()
# #     for i, val in enumerate(values):
# #         str_val = " " + str(val)
# #         if val < 1.0:
# #             str_val = " {0:.2f}".format(val)
# #         t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
# #         if i == (len(values)-1):
# #             adjust_axes(r, t, fig, axes)
# #
# #     fig.tight_layout()
# #     if plt_show:
# #         plt.show()
# #     plt.close()
# #
# #
# # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat"]
# # values = [0.2,0.3,0.4,0.5,0.6]
# # plot_title = 'plot_title'
# # x_label ='x_label'
# # draw_plot_func(values, name_classes, plot_title, x_label)#
#

# ------------------------------------------------------#
#  测试 获得子文件名字，更新train.txt文件
# ------------------------------------------------------#
import os
# # os.listdir()方法获取文件夹名字，返回数组
# def getAllFiles(targetDir):
#     listFiles = os.listdir(targetDir)
#     return listFiles
#
# files = getAllFiles(r"D:\Python_Deeplabv3p\VOCdevkit\Cityscapes\gtFine\val")
#
# s=''
# for i in files:
#     s+="'"+str(i)+"/',"
# print(s)
#
# from utils.utils import WriteTruepath_to_txt
# WriteTruepath_to_txt()

# import pickle
#
# f = open('DenseASPP_File/DenseASPP-master/denseASPP161_795.pkl','rb')
# data = pickle.load(f)
# print(data)



# ------------------------------------------------------#
#  训练中断后,加载log中参数测试
# ------------------------------------------------------#
import torch
# pthfile = 'model_data/deeplab_mobilenetv2.pth'  # .pth文件的路径
# pthfile = 'model_data/mobilenet_v2-b0353104.pth'  # .pth文件的路径
# model = torch.load(pthfile, torch.device('cuda'))
# fmt = '%-80s %-40s'
# for k in model.keys():  # 查看模型字典里面的key
#     str_temp=  str(list(model[k].shape))
#     print(fmt % (str(k), str_temp))
#
# pthfile = 'model_data/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-d256dd4b.pth'  # .pth文件的路径
# model = torch.load(pthfile, torch.device('cuda'))
# print(type(model))  # 查看模型字典长度
# print('length:')
# print(len(model))
# print('key:')
# for k in model['state_dict'].keys():  # 查看模型字典里面的key
#     print(k)

# pthfile = 'logs/last_epoch_weights.pth'  # .pth文件的路径
# model2 = torch.load(pthfile, torch.device('cuda'))
# print(type(model2))  # 查看模型字典长度
# print(len(model2))
# print('key:')
# for k in model2.keys():  # 查看模型字典里面的key
#     print(k,"  ",model2[k].shape)


# ------------------------------------------------------#
#  mobilnet pretrain参数
# ------------------------------------------------------#
# pthfile = 'DeepLabv3_MobileNetv2/MobileNetv2_DeepLabv3_cityscapes/checkpoints/Checkpoint_epoch_150.pth.tar'  # .pth文件的路径
# pthfile = 'DeepLabv3_MobileNetv2/ImageNet_pretrain.pth'  # .pth文件的路径
# pthfile = 'model_data/last_gtav_epoch_52_mean-iu_0.67255.pth'  # .pth文件的路径
# model2 = torch.load(pthfile, torch.device('cuda'))["state_dict"]
# print(type(model2))  # 查看模型字典长度
# print(len(model2))
# print('key:')
# fmt = '%-80s %-40s'
# for k in model2.keys():  # 查看模型字典里面的key
#     str_temp=  str(list(model2[k].shape))
#     print(fmt % (str(k), str_temp))

# pthfile = 'DeepLabv3_MobileNetv2/ImageNet_pretrain.pth'  # .pth文件的路径
# model2 = torch.load(pthfile, torch.device('cuda'))
# pthfile = 'logs/last_epoch_weights.pth'  # .pth文件的路径
# model3 = torch.load(pthfile, torch.device('cuda'))
# fmt = '%-40s %-40s %-70s %-40s'
# for k2,k3 in zip(model2.keys(),model3.keys()):  # 查看模型字典里面的key
#     # if k2.split('.')[0]!="backbone":
#     #     break
#     str2 = str(list(model2[k2].shape))
#     str3 = str(list(model3[k3].shape))
#     print(fmt % (str(k2), str2, str(k3), str3 ))
#     model2[k2]=model3[k3]


# pthfile ='logs/last_epoch_weights.pth'
# model = torch.load(pthfile, torch.device('cuda'))
# # pthfile = 'DeepLabv3_MobileNetv2/MobileNetv2_DeepLabv3_cityscapes/checkpoints/Checkpoint_epoch_150.pth.tar'  # .pth文件的路径
# # pthfile = 'model_data/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-d256dd4b.pth'  # .pth文件的路径
# pthfile = 'DeepLabv3_MobileNetv2/ImageNet_pretrain.pth'  # .pth文件的路径
# model3 = torch.load(pthfile, torch.device('cuda'))
#
# model2 ={}
# for k in model.keys():  # 查看模型字典里面的key
#     if k.split('.')[-1]!="num_batches_tracked":
#         model2[k] = model[k]
#

# fmt = '%-40s %-40s %-40s %-40s'
# for k2,k3 in zip(model2.keys(),model3.keys()):  # 查看模型字典里面的key
#     str2 = str(list(model2[k2].shape))
#     str3 = str(list(model3[k3].shape))
#     print(fmt % (str(k2), str2, str(k3), str3 ))

# fmt = '%-40s %-40s %-70s %-40s'
# for k2,k3 in zip(model2.keys(),model3.keys()):  # 查看模型字典里面的key
#     if k2.split('.')[0]!="backbone":
#         break
#     str2 = str(list(model2[k2].shape))
#     str3 = str(list(model3[k3].shape))
#     print(fmt % (str(k2), str2, str(k3), str3 ))
#     model2[k2]=model3[k3]

# torch.save(model2, os.path.join("model_data", "Mobilenetv2_backnone_init.pth"))

# model2 ={}
# for k in model.keys():  # 查看模型字典里面的key
#     if k.split('.')[-1]!="num_batches_tracked":
#         model2[k] = model[k]


# ------------------------------------------------------#
#  mmsegmentation
# ------------------------------------------------------#
# from mmseg.apis import inference_segmentor, init_segmentor
# import mmcv
#
# config_file = 'mmsegmentation/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes.py'
# checkpoint_file = 'mmsegmentation/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-d256dd4b.pth'
#
# # 通过配置文件和模型权重文件构建模型
# model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
#
# # 对单张图片进行推理并展示结果
# img = 'mmsegmentation/demo/demo1.png'  # or img = mmcv.imread(img), which will only load it once
# result = inference_segmentor(model, img)
# # 在新窗口中可视化推理结果
# model.show_result(img, result, show=True)
# # 或将可视化结果存储在文件中
# # 你可以修改 opacity 在(0,1]之间的取值来改变绘制好的分割图的透明度
# model.show_result(img, result, out_file='result3.jpg', opacity=0.5)


# # 对视频进行推理并展示结果
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#    result = inference_segmentor(model, frame)
#    model.show_result(frame, result, wait_time=1)

# ------------------------------------------------------#
#  查看json文件
# ------------------------------------------------------#
# import json
# import os
# import datetime
#
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader
#
# image_train_dir = "VOCdevkit/Cityscapes/image_train.txt"
# image_val_dir = "VOCdevkit/Cityscapes/image_val.txt"
# label_train_dir = "VOCdevkit/Cityscapes/label_train.txt"
# label_val_dir = "VOCdevkit/Cityscapes/label_val.txt"
# with open(image_train_dir, 'r') as f:
#     image_train_lines = f.readlines()
# with open(image_val_dir, 'r') as f:
#     image_val_lines = f.readlines()
# with open(label_train_dir, 'r') as f:
#     label_train_lines = f.readlines()
# with open(label_val_dir, 'r') as f:
#     label_val_lines = f.readlines()
#
# # 格式 xxxx_xxxx_lines = ["xx_path_1/n","xx_path_2/n"...]
# assert len(image_train_lines) == len(label_train_lines) and len(image_val_lines) == len(label_val_lines)
#
# num_train = len(image_train_lines)
# num_val = len(image_val_lines)
#
# train_lines = {'Type': 0, 'image_lines': image_train_lines, 'label_lines': label_train_lines}
# val_lines = {'Type': 1, 'image_lines': image_val_lines, 'label_lines': label_val_lines}
# num_classes = 21
# check_label = np.zeros(num_classes + 1)
# check_predict = np.zeros(num_classes + 1)

# #for ind in range(len(label_train_lines)):
# for ind in range(len(image_val_lines)):
#     check_temp = np.zeros(num_classes + 1)
#     # jpath = os.path.join("VOCdevkit/Cityscapes/gtFine/train",train_lines['label_lines'][ind].split()[0])\
#     #     .replace("labelTrainIds.png","polygons.json")
#     jpath = os.path.join("VOCdevkit/Cityscapes/gtFine/val",val_lines['label_lines'][ind].split()[0])\
#         .replace("labelTrainIds.png","polygons.json")
#
#     # 读取_json文件
#     with open(jpath, 'r') as f:
#         polygons = json.load(f)
#
#     for obj in polygons['objects']:
#         # if 'motorcycle' == obj['label'] or 'bicycle' == obj['label']:
#         if 'caravan' == obj['label'] :
#             print("第 %d 幅   "%ind, jpath)
#
#
#     if ind%50==0 :
#      print("已到 %d 幅   " % ind)

#
# json_path = 'VOCdevkit/Cityscapes/gtFine/val/frankfurt/frankfurt_000001_046272_gtFine_polygons.json'
# gt_path =  json_path.replace("polygons.json","labelTrainIds.png")
# image = Image.open(gt_path)
# array = np.array(image)
# # 读取_json文件
# with open(json_path, 'r') as f:
#     polygons = json.load(f)
# for obj in polygons['objects']:
#     if 'caravan' == obj['label'] :
#         # 输出轮廓点信息
#         for polygon in obj['polygon']:
#             x, y = polygon
#             print('Point:', x, y," Label: ",array[y][x] )


# ------------------------------------------------------#
#  寻找特定的类
# ------------------------------------------------------#
# import os
# import datetime
# from PIL import Image
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# from torch.utils.data import DataLoader
#
# image_train_dir = "VOCdevkit/Cityscapes/image_train.txt"
# image_val_dir = "VOCdevkit/Cityscapes/image_val.txt"
# label_train_dir = "VOCdevkit/Cityscapes/label_train.txt"
# label_val_dir = "VOCdevkit/Cityscapes/label_val.txt"
# with open(image_train_dir, 'r') as f:
#     image_train_lines = f.readlines()
# with open(image_val_dir, 'r') as f:
#     image_val_lines = f.readlines()
# with open(label_train_dir, 'r') as f:
#     label_train_lines = f.readlines()
# with open(label_val_dir, 'r') as f:
#     label_val_lines = f.readlines()
#
# # 格式 xxxx_xxxx_lines = ["xx_path_1/n","xx_path_2/n"...]
# assert len(image_train_lines) == len(label_train_lines) and len(image_val_lines) == len(label_val_lines)
#
# num_train = len(image_train_lines)
# num_val = len(image_val_lines)
#
# train_lines = {'Type': 0, 'image_lines': image_train_lines, 'label_lines': label_train_lines}
# val_lines = {'Type': 1, 'image_lines': image_val_lines, 'label_lines': label_val_lines}
# num_classes = 21
# check_label = np.zeros(num_classes+1)
# check_predict = np.zeros(num_classes+1)
#
# #for ind in range(len(image_train_lines)):
# for ind in range(len(image_val_lines)):
#     check_temp = np.zeros(num_classes+1)
#     #label = Image.open(os.path.join("VOCdevkit/Cityscapes/gtFine/train",train_lines['label_lines'][ind].split()[0]))
#     label = Image.open(os.path.join("VOCdevkit/Cityscapes/gtFine/val", val_lines['label_lines'][ind].split()[0]))
#     label = np.array(label)
#     label[label>=num_classes]=num_classes
#     check_temp += np.bincount(label.flatten().astype(int), minlength=num_classes+1)
#     # if check_temp[20]>0:
#     #     print("第 %d 幅   "%ind, train_lines['label_lines'][ind].split()[0])
#     #     #print("第 %d 幅   " % ind, val_lines['label_lines'][ind].split()[0])
#     check_label+=check_temp
#     if ind%50==0 :
#      print("已到 %d 幅   " % ind)
# print(check_label)

# ------------------------------------------------------#
#  损失权重
# ------------------------------------------------------#
# import  numpy as np
# # cls_weights = [0.0178, 0.1078, 0.0287, 1.0000, 0.7468, 0.5343, 3.1460,
# #                1.1864, 0.0412, 0.5661, 0.1635, 0.5388, 4.8639, 0.0937,
# #                2.4508, 2.7864, 14.5221, 27.8243, 2.8149, 6.6492, 1.5848]
# #
# # cls_weights = np.array(cls_weights, np.float32)
#
#
#
# train_weight = [3.45221953e+08,4.95587160e+07,2.00894857e+08,6.72067200e+06,
#                 7.52705300e+06,1.35652900e+07,1.81381400e+06,6.11065000e+06,
#                 1.58682635e+08,7.62589100e+06,3.07080590e+07,1.18902320e+07,
#                 1.97053700e+06,5.97593160e+07,2.76046900e+06,3.56422200e+06,
#                 5.34700000e+04,2.01209000e+05,1.03209900e+06,7.28923000e+05,
#                 6.50085300e+06,1.31685080e+08]
#
# label_weight = [2.03604895e+09,3.36030285e+08,1.25977372e+09,3.62111950e+07,
#                 4.84873470e+07,6.77718170e+07,1.15103970e+07,3.05223670e+07,
#                 8.78732742e+08,6.39647780e+07,2.21459205e+08,6.72023850e+07,
#                 7.44490300e+06,3.86502898e+08,1.47750050e+07,1.29957990e+07,
#                 2.49351700e+06,1.30142200e+06,1.28639320e+07,5.44590900e+06,
#                 2.28497640e+07,7.14638869e+08]
#
# train_weight =[(np.array(train_weight).sum()-train_weight[-1])/train_weight[0:-1]]
# train_weight = train_weight[0]/train_weight[0].sum()
#
# formatted_list = ["%.4f" % item for item in train_weight]
# output = "pre : "+"[" + ", ".join(formatted_list) + "]"
# print(output)
# print()
# median =np.median(train_weight)
# print("中位数 : ",median)
# train_weight = train_weight/median
# train_weight =train_weight.tolist()
# formatted_list = ["%.4f" % item for item in train_weight]
# output = "after : "+"[" + ", ".join(formatted_list) + "]"
# print(output)
# print()
#
# label_weight =[(np.array(label_weight).sum()-label_weight[-1])/label_weight[0:-1]]
# label_weight = label_weight[0]/label_weight[0].sum()
#
# formatted_list = ["%.4f" % item for item in label_weight]
# output = "pre : "+"[" + ", ".join(formatted_list) + "]"
# print(output)
# print()
# median =np.median(label_weight)
# print("中位数 : ",median)
# label_weight = label_weight/median
# label_weight =label_weight.tolist()
# formatted_list = ["%.4f" % item for item in label_weight]
# output = "after : "+"[" + ", ".join(formatted_list) + "]"
# print(output)
# print()

# ---------------------------#
#   读取数据集对应的txt
# ---------------------------#
# import  numpy as np
# VOCdevkit_path = 'VOCdevkit'
#
# with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
#     train_lines = f.readlines()
# with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
#     val_lines = f.readlines()
#
# tran_id = np.random.randint(0,len(train_lines),5000)
# with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/VOC_train.txt"), "w") as f:
#     for ind in tran_id:
#         f.write(train_lines[ind])
#     f.close()
#
# tran_id = np.random.randint(0,len(val_lines),500)
# with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/VOC_val.txt"), "w") as f:
#     for ind in tran_id:
#         f.write(val_lines[ind])
#     f.close()

# ---------------------------#
#   np.ndarray 和 torch.tensor
# ---------------------------#
# import torch
# import numpy as np
# a = np.random.random(size=(1,4,4))
# b = torch.tensor(a,dtype=torch.float)
# c = np.array(b)
# d = np.squeeze(c)
# print(a)
# print(b)
# print(c)
# print(d)
# print(a.shape)
# print(b.shape)
# print(c.shape)
# print(d.shape)
# ---------------------------#
#  image 不失真 resize
# ---------------------------#
# from PIL import Image
# import matplotlib.pyplot as plt
# def letterbox_image(image, size):
#     # 对图片进行resize，使图片不失真。在空缺的地方进行padding
#     iw, ih = image.size
#     w, h = size
#     scale = min(w / iw, h / ih)
#     nw = int(iw * scale)
#     nh = int(ih * scale)
#
#     image = image.resize((nw, nh), Image.BICUBIC)
#     new_image = Image.new('RGB', size, (128, 128, 128))
#     new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
#     return new_image
#
#
# img = Image.open("D:\Python_Deeplabv3p\VOCdevkit\VOC2007\JPEGImages/2007_004856.jpg")
# new_image = letterbox_image(img, [512, 512])
# new_image2 = img.resize((512, 512), Image.BICUBIC)
# plt.figure()
# wspace = 0.02
# plt.subplot(1, 3, 1)
# plt.subplots_adjust(wspace=wspace)
# plt.axis('off')
# plt.imshow(img)
# plt.subplot(1, 3, 2)
# plt.subplots_adjust(wspace=wspace)
# plt.axis('off')
# plt.imshow(new_image2)
# plt.subplot(1, 3, 3)
# plt.subplots_adjust(wspace=wspace)
# plt.axis('off')
# plt.imshow(new_image)
# plt.savefig("D:/Python_Deeplabv3p/result_compare/image_resize_compare.png", dpi=300, bbox_inches='tight')
# # plt.show()
# plt.close()

# # -*- coding: utf-8 -*-
# """
# Created on Thu Nov 15 11:04:25 2018
#
# @author: duans
# """
#
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 自定义损失函数
#
# # 1. 继承nn.Mdule
# class My_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x, y):
#         return torch.mean(torch.pow((x - y), 2))
#
#
# # 2. 直接定义函数 ， 不需要维护参数，梯度等信息
# # 注意所有的数学操作需要使用tensor完成。
# def my_mse_loss(x, y):
#     return torch.mean(torch.pow((x - y), 2))
#
#
# # 3, 如果使用 numpy/scipy的操作  可能使用nn.autograd.function来计算了
# # 要实现forward和backward函数
#
# # Hyper-parameters 定义迭代次数， 学习率以及模型形状的超参数
# input_size = 1
# output_size = 1
# num_epochs = 60
# learning_rate = 0.001
#
# # Toy dataset  1. 准备数据集
# x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
#
# y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                     [3.366], [2.596], [2.53], [1.221], [2.827],
#                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
#
# # Linear regression model  2. 定义网络结构 y=w*x+b 其中w的size [1,1], b的size[1,]
# model = nn.Linear(input_size, output_size)
#
# # Loss and optimizer 3.定义损失函数， 使用的是最小平方误差函数
# # criterion = nn.MSELoss()
# # 自定义函数1
# criterion = My_loss()
#
# # 4.定义迭代优化算法， 使用的是随机梯度下降算法
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# loss_dict = []
# # Train the model 5. 迭代训练
# for epoch in range(num_epochs):
#     # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
#     inputs = torch.from_numpy(x_train)
#     targets = torch.from_numpy(y_train)
#
#     # Forward pass  5.2 前向传播计算网络结构的输出结果
#     outputs = model(inputs)
#     # 5.3 计算损失函数
#     # loss = criterion(outputs, targets)
#
#     # 1. 自定义函数1
#     # loss = criterion(outputs, targets)
#     # 2. 自定义函数
#     loss = my_mse_loss(outputs, targets)
#     # Backward and optimize 5.4 反向传播更新参数
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # 可选 5.5 打印训练信息和保存loss
#     loss_dict.append(loss.item())
#     if (epoch + 1) % 5 == 0:
#         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
#
# # Plot the graph 画出原y与x的曲线与网络结构拟合后的曲线
# predicted = model(torch.from_numpy(x_train)).detach().numpy()
# plt.plot(x_train, y_train, 'ro', label='Original data')
# plt.plot(x_train, predicted, label='Fitted line')
# plt.legend()
# plt.show()
#
# # 画loss在迭代过程中的变化情况
# plt.plot(loss_dict, label='loss for every epoch')
# plt.legend()
# plt.show()

