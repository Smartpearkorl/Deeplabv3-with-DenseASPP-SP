import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
#---------------------------------------------------------#
#   Cityscapes数据集 写入需要的路径txt
#---------------------------------------------------------#

train_dirs = ['aachen/','bochum/','bremen/','cologne/','darmstadt/','dusseldorf/',
              'erfurt/','hamburg/','hanover/','jena/','krefeld/','monchengladbach/',
              'strasbourg/','stuttgart/','tubingen/','ulm/','weimar/','zurich/']

val_dirs = ['frankfurt/','lindau/','munster/']


image_train_path = "VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/train/"
image_val_path = "VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/val/"
label_train_path = "VOCdevkit/Cityscapes/gtFine/train/"
label_val_path = "VOCdevkit/Cityscapes/gtFine/val/"

image_train_dir ="VOCdevkit/Cityscapes/image_train.txt"
image_val_dir ="VOCdevkit/Cityscapes/image_val.txt"
label_train_dir ="VOCdevkit/Cityscapes/label_train.txt"
label_val_dir ="VOCdevkit/Cityscapes/label_val.txt"

def WriteTruepath_to_txt():
    # 获得训练集的原图名字
    with open(image_train_dir, 'w') as f:
        for son_dir in train_dirs:
            true_dir = image_train_path + son_dir
            file_names = os.listdir(true_dir)
            for file_name in file_names:
                file_name = str(son_dir)+file_name
                f.write(file_name+'\n')
        f.close()

    # 获得验证集的原图名字
    with open(image_val_dir, 'w') as f:
        for son_dir in val_dirs:
            true_dir = image_val_path + son_dir
            file_names = os.listdir(true_dir)
            for file_name in file_names:
                file_name = str(son_dir)+file_name
                f.write(file_name + '\n')
        f.close()

    # 获得训练集的标签名字
    with open(label_train_dir, 'w') as f:
        for son_dir in train_dirs:
            true_dir = label_train_path + son_dir
            file_names = os.listdir(true_dir)
            for file_name in file_names:
                if "labelTrainIds.png" in file_name:
                    file_name = str(son_dir) + file_name
                    f.write(file_name + '\n')
        f.close()

    # 获得验证集的标签名字
    with open(label_val_dir, 'w') as f:
        for son_dir in val_dirs:
            true_dir = label_val_path + son_dir
            file_names = os.listdir(true_dir)
            for file_name in file_names:
                if "labelTrainIds.png" in file_name:
                    file_name = str(son_dir) + file_name
                    f.write(file_name + '\n')
        f.close()

def ReadTruepath_from_txt():
    with open(image_train_dir, 'r') as f:
        image_train_lines = f.readlines()
    with open(image_val_dir, 'r') as f:
        image_val_lines = f.readlines()
    with open(label_train_dir, 'r') as f:
        label_train_lines = f.readlines()
    with open(label_val_dir, 'r') as f:
        label_val_lines = f.readlines()

    image_train = Image.open(image_train_path + image_train_lines[0].split()[0])
    image_val = Image.open(image_val_path + image_val_lines[0].split()[0])
    label_train = Image.open(label_train_path + label_train_lines[0].split()[0])
    label_val = Image.open(label_val_path + label_val_lines[0].split()[0])

    plt.subplot(2, 2, 1)
    plt.imshow(image_train)
    plt.subplot(2, 2, 2)
    plt.imshow(image_val)
    plt.subplot(2, 2, 3)
    plt.imshow(label_train)
    plt.subplot(2, 2, 4)
    plt.imshow(label_val)
    plt.show()

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    #图像放缩
    image   = image.resize((nw,nh), Image.BICUBIC)
    #生成灰图:512*512
    new_image = Image.new('RGB', size, (128,128,128))
    #将为放缩后的图印到灰图
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)