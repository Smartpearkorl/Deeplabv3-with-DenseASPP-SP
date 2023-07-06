# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import sys
import time

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os  # 注意要输入OS模块
from tqdm import tqdm
from deeplab import DeeplabV3


name_classes_voc = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor", "ignored"]

name_classes_city = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                     "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                     "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle", "ignored"]

name_classes19_city = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                       "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                       "truck", "bus", "caravan", "motorcycle", "bicycle", "ignored"]

colors_voc = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]

colors_city = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
              (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
              (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (125, 0, 90), (0, 255, 110),
              (0, 80, 100), (0, 0, 230), (119, 11, 32), (128, 128, 128)]

colors_classes19_city = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                         (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (128, 128, 128)]

cow_names = ['2007_000464', '2007_000491', '2007_001299', '2007_002387', '2007_002903', '2007_003201', '2007_003841',
             '2007_005114', '2007_005547', '2007_006761', '2007_006841', '2007_008722', '2007_008973', '2007_009245',
             '2007_009897', '2008_000009', '2008_000073', '2008_000919', '2008_002778', '2008_004701', '2008_005097',
             '2008_005105', '2008_006063', '2008_006528', '2008_007025', '2008_007031', '2008_007123', '2008_007273',
             '2008_007596', '2009_000219', '2009_000309', '2009_000641', '2009_000731', '2009_000771', '2009_002035',
             '2009_002122', '2009_002150', '2009_002171', '2009_002221', '2009_002549', '2009_002635', '2009_003059',
             '2009_003542', '2009_003569', '2009_003773', '2010_000427', '2010_000907', '2010_001000', '2010_001010',
             '2010_001692', '2010_002390', '2010_002701', '2010_002763', '2010_003183', '2010_003239', '2010_004322',
             '2010_004635', '2010_004789', '2010_005166', '2010_005922', '2011_000548', '2011_000566', '2011_001047',
             '2011_001159', '2011_001232', '2011_001530', '2011_001546', '2011_001708', '2011_001782', '2011_002885',
             '2011_003019']

sofa_names = ['2007_000452', '2007_000661', '2007_000804', '2007_001154', '2007_001457', '2007_001458', '2007_001585',
              '2007_001763', '2007_002268', '2007_002426', '2007_002427', '2007_003011', '2007_003169', '2007_003530',
              '2007_006171', '2007_006373', '2007_007417', '2007_007996', '2007_008260', '2007_008543', '2007_009068',
              '2007_009252', '2007_009521', '2007_009592', '2007_009655', '2007_009684', '2008_000149', '2008_000270',
              '2008_000589', '2008_002929', '2008_003709', '2008_003733', '2008_004172', '2008_004562', '2008_006784',
              '2008_008103', '2009_000087', '2009_000096', '2009_000242', '2009_000418', '2009_000487', '2009_000732',
              '2009_001644', '2009_001731', '2009_001775', '2009_002346', '2009_002390', '2009_002571', '2009_002732',
              '2009_003123', '2009_003564', '2009_003607', '2009_003895', '2009_004099', '2009_004140', '2009_004298',
              '2009_004687', '2009_004993', '2010_000256', '2010_000284', '2010_001149', '2010_001327', '2010_001767',
              '2010_001962', '2010_002137', '2010_002531', '2010_003014', '2010_003597', '2010_004757', '2010_004783',
              '2010_005180', '2010_005606', '2010_005991', '2011_000070', '2011_000173', '2011_000310', '2011_001005',
              '2011_001567', '2011_001601', '2011_001674', '2011_001713', '2011_001794', '2011_001862', '2011_002124',
              '2011_002391', '2011_002644', '2011_002879', '2011_003103', '2011_003182', '2011_003256']

diningtable_names = ['2007_000762', '2007_000830', '2007_000847', '2007_001430', '2007_001677', '2007_002624', '2007_003011',
                     '2007_003742', '2007_004405', '2007_004712', '2007_005803', '2007_005844', '2007_006086', '2007_006241',
                     '2007_007007', '2007_007498', '2007_007651', '2007_007810', '2007_009413', '2007_009521', '2007_009706',
                     '2008_000573', '2008_000763', '2008_002379', '2008_002775', '2008_002864', '2008_003477', '2008_004140',
                     '2008_004575', '2008_005145', '2008_006008', '2008_006108', '2008_006408', '2008_007048', '2008_007378',
                     '2008_007402', '2008_008051', '2008_008335', '2008_008362', '2009_001215', '2009_002415', '2009_002487',
                     '2009_002571', '2009_003003', '2009_003071', '2009_003123', '2009_003507', '2009_003849', '2009_003991',
                     '2009_004140', '2009_004859', '2009_005038', '2010_000174', '2010_001579', '2010_001851', '2010_002336',
                     '2010_003325', '2010_003531', '2010_003597', '2010_003971', '2010_004825', '2010_005046', '2010_005626',
                     '2011_000226', '2011_000338', '2011_000598', '2011_000809', '2011_000953', '2011_001069', '2011_001276',
                     '2011_001665', '2011_001862', '2011_002075', '2011_002200', '2011_002358']

caravan_names = ['frankfurt/frankfurt_000001_046272_gtFine_labelTrainIds.png',
                 'frankfurt/frankfurt_000001_046504_gtFine_labelTrainIds.png',
                 'munster/munster_000019_000019_gtFine_labelTrainIds.png',
                 'munster/munster_000020_000019_gtFine_labelTrainIds.png',
                 'munster/munster_000098_000019_gtFine_labelTrainIds.png',
                 'munster/munster_000099_000019_gtFine_labelTrainIds.png',
                 'munster/munster_000124_000019_gtFine_labelTrainIds.png',
                 'munster/munster_000171_000019_gtFine_labelTrainIds.png']

caravan_mapi_names = ['D6cXkCE002CQwE5x6sDt4w.png', 'iU9Fvr7Cn05Db0hbktR02g.png', 'iYGUOkS9D57bpnPOLgf_7A.png', 'JYcWwc0EXi0T5tcwYJwWGg.png',
                      'kqxMTbTPAG4kG1U1i1dPVw.png', 'P2COoyHFwi_WiFhVn-QcJQ.png', 'soimOwA7vWWoTwE8jCOr5w.png', 'YOYnjnRGsDuKu1ohMngKRA.png',
                      '_EGt7GTUW6OKw0zzz7c5-A.png', '_k0fa6yqavqlHEtN_qT4Rw.png']

aspp_save_path = "./result_compare/pred_aspp_voc"
dense_save_path = './result_compare/pred_dense_voc'
gt_save_path = './result_compare/gt_voc'

aspp_city_save_path = "./result_compare/pred_aspp_city"
dense_city_save_path = './result_compare/pred_dense_city'
gt_city_save_path = './result_compare/gt_city'

aspp_mapi_save_path = "./result_compare/pred_aspp_mapi"
dense_mapi_save_path = './result_compare/pred_dense_mapi'
gt_mapi_save_path = './result_compare/gt_mapi'

iwl_city_pred_path = "./result_compare/palette_iwl_pred_city"
iwl_mapi_pred_path = "./result_compare/palette_iwl_pred_mapi"
iwl_city_save_path = './result_compare/pred_iwl_city'
iwl_mapi_save_path = './result_compare/pred_mapi_city'

plot_save_path = "./result_compare/plot_img"
plot_plus_save_path ="./result_compare/plot_plus_img"

def Get_Class_ImgNames(Type="voc",search_class="sofa"):
    if Type == "voc":
        VOCdevkit_path = 'VOCdevkit'
        with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
            val_lines = f.readlines()

        num_classes = 21

        ind = name_classes_voc.index(search_class)
        search_class_id = []

        for line in tqdm(val_lines):
            check_temp = np.zeros(num_classes + 1)
            label = Image.open(os.path.join("VOCdevkit/VOC2007/SegmentationClass", line.split()[0] + ".png"))
            label = np.array(label)
            label[label >= num_classes] = num_classes
            check_temp += np.bincount(label.flatten().astype(int), minlength=num_classes + 1)
            if check_temp[ind] > 0:
                search_class_id.append(line.split()[0])
        print(search_class_id)
        return search_class_id

    elif Type == "city":
        image_val_dir = "VOCdevkit/Cityscapes/image_val.txt"
        label_val_dir = "VOCdevkit/Cityscapes/label_val.txt"


        with open(label_val_dir, 'r') as f:
            label_val_lines = f.readlines()

        num_classes = 21
        check_label = np.zeros(num_classes + 1)
        check_predict = np.zeros(num_classes + 1)

        ind = name_classes_city.index(search_class)
        search_class_id = []

        for line in tqdm(label_val_lines):
            check_temp = np.zeros(num_classes + 1)
            label = Image.open(os.path.join("VOCdevkit/Cityscapes/gtFine/val", line.split()[0]))
            label = np.array(label)
            label[label >= num_classes] = num_classes
            check_temp += np.bincount(label.flatten().astype(int), minlength=num_classes + 1)
            if check_temp[ind] > 0:
                search_class_id.append(line.split()[0])
            check_label += check_temp
        print(search_class_id)
        return search_class_id

    elif Type == "mapi":

        label_val_dir = "VOCdevkit/Mapilary/label_val.txt"
        with open(label_val_dir, 'r') as f:
            label_val_lines = f.readlines()

        num_classes = 21
        check_label = np.zeros(num_classes + 1)
        check_predict = np.zeros(num_classes + 1)

        ind = name_classes_city.index(search_class)
        search_class_id = []

        for line in tqdm(label_val_lines):
            check_temp = np.zeros(num_classes + 1)
            label = Image.open(os.path.join("./VOCdevkit/Mapilary/validation/class21_labels", line.split()[0]+".png"))
            label = np.array(label)
            label[label >= num_classes] = num_classes
            check_temp += np.bincount(label.flatten().astype(int), minlength=num_classes + 1)
            if check_temp[ind] > 0:
                search_class_id.append(line.split()[0]+".png")
            check_label += check_temp
        print(search_class_id)
        return search_class_id
    else:
        raise ValueError("Wrong type : {}").format(Type)

def Get_pred_img(Type="voc",img_names=sofa_names):
    if Type=="voc":
        if not os.path.exists(aspp_save_path):
            os.makedirs(aspp_save_path)
        if not os.path.exists(dense_save_path):
            os.makedirs(dense_save_path)
        if not os.path.exists(gt_save_path):
            os.makedirs(gt_save_path)

        net_aspp = DeeplabV3(type="aspp_voc")
        net_dense = DeeplabV3(type="dense_voc")
        for img_name in tqdm(img_names):
            img = Image.open(os.path.join("VOCdevkit/VOC2007/JPEGImages", img_name + ".jpg"))
            pred_aspp = net_aspp.detect_image(img)
            pred_dense = net_dense.detect_image(img)
            pred_aspp.save(os.path.join(aspp_save_path, img_name + ".png"))
            pred_dense.save(os.path.join(dense_save_path, img_name + ".png"))

            gt = Image.open(os.path.join("VOCdevkit/VOC2007/SegmentationClass", img_name + ".png"))
            gt_array = np.array(gt)
            orininal_h = gt_array.shape[0]
            orininal_w = gt_array.shape[1]
            gt_array[gt_array >= 21] = 21
            colors = colors_voc
            color_np = np.array(colors, np.uint8)
            ss = np.reshape(gt_array, [-1])
            seg_img = np.reshape(color_np[ss], [orininal_h, orininal_w, -1])
            gt_color = Image.fromarray(np.uint8(seg_img))
            gt_color.save((os.path.join(gt_save_path, img_name + ".png")))
    elif Type == "city":
        if not os.path.exists(aspp_city_save_path):
            os.makedirs(aspp_city_save_path)
        if not os.path.exists(dense_city_save_path):
            os.makedirs(dense_city_save_path)
        if not os.path.exists(gt_city_save_path):
            os.makedirs(gt_city_save_path)
        net_aspp = DeeplabV3(type="aspp_city")
        net_dense = DeeplabV3(type="dense_city")
        for img_name in tqdm(img_names):
            img_name_origin=img_name.replace("gtFine_labelTrainIds","leftImg8bit")
            img = Image.open(os.path.join("VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/val",img_name_origin))
            pred_aspp = net_aspp.detect_image(img)
            pred_dense = net_dense.detect_image(img)
            # frankfurt / frankfurt_000000_000294_gtFine_labelTrainIds.png
            save_name=img_name.split("/")[1].replace("_gtFine_labelTrainIds","")
            pred_aspp.save(os.path.join(aspp_city_save_path, save_name))
            pred_dense.save(os.path.join(dense_city_save_path, save_name))

            gt = Image.open(os.path.join("VOCdevkit/Cityscapes/gtFine/val", img_name))
            gt_array = np.array(gt)
            orininal_h = gt_array.shape[0]
            orininal_w = gt_array.shape[1]
            gt_array[gt_array >= 21] = 21
            colors = colors_city
            color_np = np.array(colors, np.uint8)
            ss = np.reshape(gt_array, [-1])
            seg_img = np.reshape(color_np[ss], [orininal_h, orininal_w, -1])
            gt_color = Image.fromarray(np.uint8(seg_img))
            gt_color.save((os.path.join(gt_city_save_path, save_name)))

    elif Type == "mapi":
        if not os.path.exists(aspp_mapi_save_path):
            os.makedirs(aspp_mapi_save_path)
        if not os.path.exists(dense_mapi_save_path):
            os.makedirs(dense_mapi_save_path)
        if not os.path.exists(gt_mapi_save_path):
            os.makedirs(gt_mapi_save_path)
        net_aspp = DeeplabV3(type="aspp_city")
        net_dense = DeeplabV3(type="dense_city")

        for img_name in tqdm(img_names):
            img_name_origin=img_name.replace(".png",".jpg")
            img = Image.open(os.path.join("./VOCdevkit/Mapilary/validation/images",img_name_origin))
            pred_aspp = net_aspp.detect_image(img)
            pred_dense = net_dense.detect_image(img)
            save_name=img_name
            pred_aspp.save(os.path.join(aspp_mapi_save_path, save_name))
            pred_dense.save(os.path.join(dense_mapi_save_path, save_name))

            gt = Image.open(os.path.join("./VOCdevkit/Mapilary/validation/class21_labels", img_name))
            gt_array = np.array(gt)
            orininal_h = gt_array.shape[0]
            orininal_w = gt_array.shape[1]
            gt_array[gt_array >= 21] = 21
            colors = colors_city
            color_np = np.array(colors, np.uint8)
            ss = np.reshape(gt_array, [-1])
            seg_img = np.reshape(color_np[ss], [orininal_h, orininal_w, -1])
            gt_color = Image.fromarray(np.uint8(seg_img))
            gt_color.save((os.path.join(gt_mapi_save_path, save_name)))
    else:
        raise ValueError("Wrong type : {}").format(Type)

def Get_robust_predColor(Type="city"):
    if Type == "city":
        from cityscapesScripts_master.cityscapesscripts.helpers import labels
        pred_path = iwl_city_pred_path
        save_path = iwl_city_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img_names = os.listdir(pred_path)
        for img_name in tqdm(img_names):
            save_name = img_name.replace("_leftImg8bit", "")
            gt = Image.open(os.path.join(pred_path, img_name))
            gt_array = np.array(gt)
            label_out = np.zeros_like(gt_array)
            for id, label in labels.id2label.items():
                label_out[np.where(gt_array == id)] = label.trainId
            gt_array = label_out
            orininal_h = gt_array.shape[0]
            orininal_w = gt_array.shape[1]
            gt_array[gt_array >= 19] = 19
            colors = colors_classes19_city
            color_np = np.array(colors, np.uint8)
            ss = np.reshape(gt_array, [-1])
            seg_img = np.reshape(color_np[ss], [orininal_h, orininal_w, -1])
            gt_color = Image.fromarray(np.uint8(seg_img))
            gt_color.save(os.path.join(save_path, save_name))

    elif Type == "mapi":
        pred_path = iwl_mapi_pred_path
        save_path = iwl_mapi_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_names = os.listdir(pred_path)

        for img_name in tqdm(img_names):
            save_name = img_name
            gt = Image.open(os.path.join("./VOCdevkit/Mapilary/validation/class21_labels", img_name))
            gt_array = np.array(gt)
            orininal_h = gt_array.shape[0]
            orininal_w = gt_array.shape[1]
            pred = Image.open(os.path.join(pred_path, img_name))
            pred_array = np.array(pred)
            pred_array = cv2.resize(pred_array, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pred_array[pred_array >= 19] = 19
            colors = colors_classes19_city
            color_np = np.array(colors, np.uint8)
            ss = np.reshape(pred_array, [-1])
            seg_img = np.reshape(color_np[ss], [orininal_h, orininal_w, -1])
            pre_color = Image.fromarray(np.uint8(seg_img))
            pre_color.save(os.path.join(save_path, save_name))

    else:
        raise ValueError("Wrong type : {}").format(Type)

def Get_compare_plot(Type="voc",class_name="unnamed",img_names=None):

    class_plot_save_path=os.path.join(plot_save_path,class_name)
    if not os.path.exists(class_plot_save_path):
        os.makedirs(class_plot_save_path)

    for img_name in tqdm(img_names):
        if Type == "voc":
            img = Image.open(os.path.join("VOCdevkit/VOC2007/JPEGImages", img_name + ".jpg"))
            gt =  Image.open(os.path.join(gt_save_path, img_name + ".png"))
            pred_aspp = Image.open(os.path.join(aspp_save_path, img_name + ".png"))
            pred_dense = Image.open(os.path.join(dense_save_path, img_name + ".png"))
            save_name=img_name + ".png"

        elif Type == "city":
            img_name_origin = img_name.replace("gtFine_labelTrainIds", "leftImg8bit")
            img = Image.open(os.path.join("VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/val", img_name_origin))
            save_name = img_name.split("/")[1].replace("_gtFine_labelTrainIds", "")
            gt =  Image.open(os.path.join(gt_city_save_path, save_name))
            pred_aspp = Image.open(os.path.join(aspp_city_save_path, save_name))
            pred_dense = Image.open(os.path.join(dense_city_save_path, save_name))

        plt.figure()
        wspace=0.02
        plt.subplot(1, 4, 1)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(img)
        plt.subplot(1, 4, 2)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(pred_aspp)
        plt.subplot(1, 4, 3)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(pred_dense)
        plt.subplot(1, 4, 4)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(gt)

        plt.savefig(os.path.join(class_plot_save_path,save_name),dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

def Get_compare_plot_plus(Type="voc",class_name="unnamed",img_names=None):

    class_plot_save_path=os.path.join(plot_plus_save_path,class_name)
    if not os.path.exists(class_plot_save_path):
        os.makedirs(class_plot_save_path)

    for img_name in tqdm(img_names):
        if Type == "voc":
            img = Image.open(os.path.join("VOCdevkit/VOC2007/JPEGImages", img_name + ".jpg"))
            gt =  Image.open(os.path.join(gt_save_path, img_name + ".png"))
            pred_aspp = Image.open(os.path.join(aspp_save_path, img_name + ".png"))
            pred_dense = Image.open(os.path.join(dense_save_path, img_name + ".png"))
            save_name=img_name + ".png"

        elif Type == "city":
            img_name_origin = img_name.replace("gtFine_labelTrainIds", "leftImg8bit")
            img = Image.open(os.path.join("VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/val", img_name_origin))
            save_name = img_name.split("/")[1].replace("_gtFine_labelTrainIds", "")
            gt =  Image.open(os.path.join(gt_city_save_path, save_name))
            pred_aspp = Image.open(os.path.join(aspp_city_save_path, save_name))
            pred_dense = Image.open(os.path.join(dense_city_save_path, save_name))
            pred_iwl = Image.open(os.path.join(iwl_city_save_path, save_name))

        elif Type == "mapi":
            img_name_origin = img_name.replace(".png",".jpg")
            img = Image.open(os.path.join("./VOCdevkit/Mapilary/validation/images", img_name_origin))
            save_name = img_name
            gt =  Image.open(os.path.join(gt_mapi_save_path, save_name))
            pred_aspp = Image.open(os.path.join(aspp_mapi_save_path, save_name))
            pred_dense = Image.open(os.path.join(dense_mapi_save_path, save_name))
            pred_iwl = Image.open(os.path.join(iwl_mapi_save_path, save_name))

        plt.figure()
        wspace=0.02
        plt.subplot(1, 5, 1)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(img)
        plt.subplot(1, 5, 2)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(pred_aspp)
        plt.subplot(1, 5, 3)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(pred_dense)
        plt.subplot(1, 5, 4)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(pred_iwl)
        plt.subplot(1, 5, 5)
        plt.subplots_adjust(wspace=wspace)
        plt.axis('off')
        plt.imshow(gt)

        plt.savefig(os.path.join(class_plot_save_path,save_name),dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

def Get_rectangle_color_map(Type="voc"):
    re_h = 25
    re_w = 60
    rectangle = np.zeros((re_h, re_w ),dtype=np.uint8)
    rectangle_list = []
    for i in range(1,22):
        rectangle_list.append(rectangle+i)

    rectangle_map_1 = rectangle
    rectangle_map_2 = rectangle_list[10]
    for i in range(0, 10):
        rectangle_map_1= np.concatenate((rectangle_map_1, rectangle_list[i]),axis=1)
        rectangle_map_2= np.concatenate((rectangle_map_2, rectangle_list[i+11]),axis=1)

    rectangle_map=np.concatenate((rectangle_map_1,rectangle_map_2),axis=0)

    orininal_h = rectangle_map.shape[0]
    orininal_w = rectangle_map.shape[1]

    if Type=="voc":
        colors = colors_voc
        name_classes = name_classes_voc

    elif Type=="city":
        colors = colors_city
        name_classes = name_classes_city

    else:
        raise ValueError("Wrong type : {}").format(Type)

    color_np = np.array(colors, np.uint8)
    ss = np.reshape(rectangle_map, [-1])
    seg_img = np.reshape(color_np[ss], [orininal_h, orininal_w, -1])
    color_map = Image.fromarray(np.uint8(seg_img))
    plt.figure(dpi=600)
    plt.axis('off')
    plt.imshow(color_map)

    for y in range(2):
        for x in range(11):
            plt.text( x*re_w+10 , y*re_h+12,name_classes[x+y*11] , color="white", fontsize=4, va='center', fontweight='bold')

    plt.savefig(os.path.join("./result_compare", Type+"_color_map" + ".png"), dpi=300, bbox_inches='tight')
    plt.show()

def Get_rectangle_color_map_plus():
    re_h = 25
    re_w = 60
    rectangle = np.zeros((re_h, re_w ),dtype=np.uint8)
    rectangle_list = []
    for i in range(1,20):
        rectangle_list.append(rectangle+i)

    rectangle_map_1 = rectangle
    rectangle_map_2 = rectangle_list[9]
    for i in range(0, 9):
        rectangle_map_1= np.concatenate((rectangle_map_1, rectangle_list[i]),axis=1)
        rectangle_map_2= np.concatenate((rectangle_map_2, rectangle_list[i+10]),axis=1)

    rectangle_map=np.concatenate((rectangle_map_1,rectangle_map_2),axis=0)

    orininal_h = rectangle_map.shape[0]
    orininal_w = rectangle_map.shape[1]


    colors = colors_classes19_city

    name_classes = name_classes19_city

    color_np = np.array(colors, np.uint8)
    ss = np.reshape(rectangle_map, [-1])
    seg_img = np.reshape(color_np[ss], [orininal_h, orininal_w, -1])
    color_map = Image.fromarray(np.uint8(seg_img))
    plt.figure(dpi=600)
    plt.axis('off')
    plt.imshow(color_map)

    for y in range(2):
        for x in range(10):
            plt.text( x*re_w+10 , y*re_h+12,name_classes[x+y*10] , color="white", fontsize=4, va='center', fontweight='bold')

    plt.savefig(os.path.join("./result_compare", "class19_color_map" + ".png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # img_names=Get_Class_ImgNames("voc","sofa")
    # img_names2=Get_Class_ImgNames("voc","cow")
    # img_names3=Get_Class_ImgNames("voc","diningtable")
    # Get_pred_img("voc",img_names)
    # Get_pred_img("voc",img_names2)
    # Get_pred_img("voc",img_names3)


    # Get_robust_predColor("city")
    # Get_robust_predColor("mapi")

    # img_names = Get_Class_ImgNames("mapi","caravan")
    img_names = os.listdir("result_compare/gt_mapi")
    # Get_pred_img("mapi", img_names)
    img_names = img_names[100:]
    Get_compare_plot_plus("mapi", "all_class", img_names)

    # img_names=Get_Class_ImgNames("city","caravan")
    # img_names2=Get_Class_ImgNames("city","terrain")
    # img_names3=Get_Class_ImgNames("city","trailer")
    #
    # # # Get_pred_img("city",img_names)
    # # # Get_pred_img("city",img_names2)
    # # # Get_pred_img("city",img_names3)
    #
    # # Get_compare_plot("city", "caravan", img_names)
    # # Get_compare_plot("city", "terrain", img_names2)
    # # Get_compare_plot("city", "trailer", img_names3)
    #
    # Get_compare_plot_plus("city", "caravan", img_names)
    # Get_compare_plot_plus("city", "terrain", img_names2)
    # Get_compare_plot_plus("city", "trailer", img_names3)

    # Get_rectangle_color_map("voc")
    # Get_rectangle_color_map("city")
    # Get_rectangle_color_map_plus()
