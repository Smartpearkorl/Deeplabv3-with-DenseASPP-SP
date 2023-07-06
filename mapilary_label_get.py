

import json
import os
import numpy as np
from PIL import Image, ImageCms

import matplotlib.pyplot as plt
# Convert this dataset to have labels from cityscapes
num_classes = 21
ignore_label = 255
id_to_ignore_or_group = {}

def gen_id_to_ignore():
    global id_to_ignore_or_group
    for i in range(66):
        id_to_ignore_or_group[i] = ignore_label

    ### Convert each class to cityscapes one
    ### Road
    # Road
    id_to_ignore_or_group[13] = 0
    # Lane Marking - General
    id_to_ignore_or_group[24] = 0
    # Manhole
    id_to_ignore_or_group[41] = 0

    ### Sidewalk
    # Curb
    id_to_ignore_or_group[2] = 1
    # Sidewalk
    id_to_ignore_or_group[15] = 1

    ### Building
    # Building
    id_to_ignore_or_group[17] = 2

    ### Wall
    # Wall
    id_to_ignore_or_group[6] = 3

    ### Fence
    # Fence
    id_to_ignore_or_group[3] = 4

    ### Pole
    # Pole
    id_to_ignore_or_group[45] = 5
    # Utility Pole
    id_to_ignore_or_group[47] = 5

    ### Traffic Light
    # Traffic Light
    id_to_ignore_or_group[48] = 6

    ### Traffic Sign
    # Traffic Sign
    id_to_ignore_or_group[50] = 7

    ### Vegetation
    # Vegitation
    id_to_ignore_or_group[30] = 8

    ### Terrain
    # Terrain
    id_to_ignore_or_group[29] = 9

    ### Sky
    # Sky
    id_to_ignore_or_group[27] = 10

    ### Person
    # Person
    id_to_ignore_or_group[19] = 11

    ### Rider
    # Bicyclist
    id_to_ignore_or_group[20] = 12
    # Motorcyclist
    id_to_ignore_or_group[21] = 12
    # Other Rider
    id_to_ignore_or_group[22] = 12

    ### Car
    # Car
    id_to_ignore_or_group[55] = 13

    ### Truck
    # Truck
    id_to_ignore_or_group[61] = 14

    ### Bus
    # Bus
    id_to_ignore_or_group[54] = 15

    ### caravan
    # caravan
    id_to_ignore_or_group[56] = 16

    ### trailer
    # trailer
    id_to_ignore_or_group[60] = 17

    ### train
    # On Rails
    id_to_ignore_or_group[58] = 18

    ### Motorcycle
    # Motorcycle
    id_to_ignore_or_group[57] = 19

    ### Bicycle
    # Bicycle
    id_to_ignore_or_group[52] = 20

def Demo_show_one_pic(path):
    mask_path=path+".jpg"
    image = Image.open(mask_path)
    mask_path=path+".png"
    mask = Image.open(mask_path)
    img_name = os.path.splitext(os.path.basename(mask_path))[0]
    shape = mask.size
    mask = np.array(mask)
    mask_copy = mask.copy()
    gen_id_to_ignore()
    for k, v in id_to_ignore_or_group.items():
        mask_copy[mask == k] = v

    mask_copy[mask_copy >=num_classes] = num_classes
    colors = [ (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30),
            (220,220,  0), (107,142, 35), (152,251,152),( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
            (  0, 60,100), (  0,  0, 90), (  0,  0,110),(  0, 80,100), (  0,  0,230), (119, 11, 32), (128 , 128 , 128)]
    color_np = np.array(colors, np.uint8)
    ss = np.reshape(mask_copy, [-1])
    seg_img = np.reshape(color_np[ss], [shape[1],shape[0] , -1])
    seg_img   = Image.fromarray(np.uint8(seg_img))
    image3 = Image.blend(image, seg_img, 0.5)

    array = np.array(image)
    array2 = np.array(seg_img)
    array3=np.array(image3)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(array)
    plt.subplot(1,3,2)
    plt.imshow(array2)
    plt.subplot(1,3,3)
    plt.imshow(array3)
    plt.show()


image_train_path = "VOCdevkit/Mapilary/training/images"
image_val_path = "VOCdevkit/Mapilary/validation/images"
label_train_path = "VOCdevkit/Mapilary/training/labels"
label_val_path = "VOCdevkit/Mapilary/validation/labels"

new_label_train_path = "VOCdevkit/Mapilary/training/class21_labels"
new_label_val_path = "VOCdevkit/Mapilary/validation/class21_labels"

train_dir ="VOCdevkit/Mapilary/label_train.txt"
val_dir ="VOCdevkit/Mapilary/label_val.txt"

def show_one_label_resault(label_name):
    path = os.path.join(image_train_path,label_name + ".jpg")
    image = Image.open(path)
    mask_path = os.path.join(new_label_train_path,label_name + ".png")
    mask = Image.open(mask_path)
    shape = mask.size
    mask = np.array(mask)
    mask[mask >= num_classes] = num_classes
    colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
              (250, 170, 30),
              (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (128, 128, 128)]
    color_np = np.array(colors, np.uint8)
    ss = np.reshape(mask, [-1])
    seg_img = np.reshape(color_np[ss], [shape[1], shape[0], -1])
    seg_img = Image.fromarray(np.uint8(seg_img))
    image3 = Image.blend(image, seg_img, 0.5)
    array = np.array(image)
    array2 = np.array(seg_img)
    array3 = np.array(image3)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(array)
    plt.subplot(1, 3, 2)
    plt.imshow(array2)
    plt.subplot(1, 3, 3)
    plt.imshow(array3)
    plt.show()

# with open(train_dir, 'r') as f:
#     val_lines = f.readlines()
# show_one_label_resault(val_lines[0].split()[0])

def Get_Class21_label():

    if not os.path.exists(new_label_train_path):
        os.makedirs(new_label_train_path)
    if not os.path.exists(new_label_val_path):
        os.makedirs(new_label_val_path)
    gen_id_to_ignore()
    with open(val_dir, 'w') as f:
        count=0
        file_names = os.listdir(label_val_path)
        for file_name in file_names:
            mask_path = os.path.join(label_val_path,file_name)
            mask = Image.open(mask_path)
            img_name = os.path.splitext(os.path.basename(mask_path))[0]
            mask = np.array(mask)
            mask_copy = mask.copy()
            for k, v in id_to_ignore_or_group.items():
                mask_copy[mask == k] = v
            mask = Image.fromarray(mask_copy.astype(np.uint8))
            mask.save(os.path.join(new_label_val_path, img_name+'.png'))
            f.write(img_name+'\n')
            count=count+1
            if(count%50==0):
                print("val count = %d"%count)
        f.close()

    # with open(train_dir, 'w') as f:
    #     count=0
    #     file_names = os.listdir(label_train_path)
    #     for file_name in file_names:
    #         mask_path = os.path.join(label_train_path,file_name)
    #         mask = Image.open(mask_path)
    #         img_name = os.path.splitext(os.path.basename(mask_path))[0]
    #         mask = np.array(mask)
    #         mask_copy = mask.copy()
    #         for k, v in id_to_ignore_or_group.items():
    #             mask_copy[mask == k] = v
    #         mask = Image.fromarray(mask_copy.astype(np.uint8))
    #         mask.save(os.path.join(new_label_train_path, img_name+'.png'))
    #         f.write(img_name+'\n')
    #         count=count+1
    #         if (count >5000 ):
    #             break
    #         if(count%50):
    #             print("count = %d"%count)
    #     f.close()

Get_Class21_label()


