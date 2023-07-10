
<font color=blue > </font>
<font color=cyan >**颜色**</font>
<font color= blueviolet>**颜色**</font>
<font color= chartreuse>**颜色**</font>
<font color= crismon>**颜色**</font>
<font color= darkcyan>**颜色**</font>
<font color= darkorange>**颜色**</font>
<font color= deeppink>**颜色**</font>
<font color=gold >**颜色**</font>
<font color= fuchsia>**颜色**</font>
***  
* * *  
*****
- - -
----------
> 最外层
> 
> 第一层嵌套
> > > 第二层嵌套
* 第一项
    > 菜鸟教程
    > 学的不仅是技术更是梦想
* 第二项
* 第一项
* 第二项
* 第三项

+ 第一项
+ 第二项
+ 第三项

- 第一项
- 第二项
- 第三项

1. 第一项
2. 第二项
3. 第三项



## DeepLabv3+ with DenseASPP and Strip pooling
---
### 项目说明  
       采用Pytorch框架。整体采用了以Deeplabv2为backbone的Deeplabv3+网络，并在此基础上，引入DenseASPP & Strip Pooling, 在PASCALVOC & CityScapes 数据集进行训练和验证。结果对比发现，改进后的模型对比原模型有更加优越。
### 目录
1. [News Update](#news_update)
2. [模型训练 Training](#模型训练)
3. 第三项
### News_Update
**`2023-07-10`**:**第二个下午，希望能把训练代码弄完**   
**`2023-07-06`**:**开始整理毕设代码，希望能弄完** 
### <span id="jump_性能情况">性能情况</span> 
|network | train dataset | weight file | val dataset | input size | mIOU | 
| :-----:| :-----: | :-----: | :------: | :------: | :------: | 
|ASPP | VOC | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | VOC-Val | 512x512| 72.63 | 
|DenseASPP+SP | VOC | [deeplab_mobilenetv2.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_mobilenetv2.pth) | VOC-Val | 512x512| 74.44 | 
|ASPP | Cityscapes| [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth) | Cityscapes-Val | 512x512|64.92 | 
|DenseASPP+SP | Cityscapes| [deeplab_xception.pth](https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/deeplab_xception.pth) | Cityscapes-Val | 512x512| 67.37 | 
### 模型训练
训练代码没有整合，分为四个文件，对应[性能情况](#jump_性能情况)的四个mIOU结果。具体在代码中，也就是网络结构和数据集不同。
```python
#网络结构
from nets.deeplabv3_plus import DeepLab # Network: ASPP
from nets.deeplabv3_plus_DenseAspp import DeepLab_Dense # Network: DenseASPP+SP

#数据集
#---------------------------#
#   读取VOC数据集对应的txt
#---------------------------#
with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
    train_lines = f.readlines()
with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
    val_lines = f.readlines()

# ---------------------------#
#   Cityscapes数据集构造需要的txt并读取，读取数据集对应的txt
# ---------------------------#
image_train_dir = "VOCdevkit/Cityscapes/image_train.txt"
image_val_dir = "VOCdevkit/Cityscapes/image_val.txt"
label_train_dir = "VOCdevkit/Cityscapes/label_train.txt"
label_val_dir = "VOCdevkit/Cityscapes/label_val.txt"
with open(image_train_dir, 'r') as f:
    image_train_lines = f.readlines()
with open(image_val_dir, 'r') as f:
    image_val_lines = f.readlines()
with open(label_train_dir, 'r') as f:
    label_train_lines = f.readlines()
with open(label_val_dir, 'r') as f:
    label_val_lines = f.readlines()

# 格式 xxxx_xxxx_lines = ["xx_path_1/n","xx_path_2/n"...]
assert len(image_train_lines)==len(label_train_lines) and len(image_val_lines)==len(label_val_lines)

num_train = len(image_train_lines)
num_val = len(image_val_lines)

train_lines = {'Type':0,'image_lines':image_train_lines,'label_lines':label_train_lines}
val_lines = {'Type':1,'image_lines':image_val_lines,'label_lines':label_val_lines}
```




