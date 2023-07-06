import os

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 21
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                    "traffic sign","vegetation","terrain", "sky", "person", "rider", "car",
                    "truck", "bus", "caravan" , "trailer", "train","motorcycle", "bicycle"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    image_train_path = "VOCdevkit/Mapilary/training/images"
    image_val_path = "VOCdevkit/Mapilary/validation/images"
    label_train_path = "VOCdevkit/Mapilary/training/labels"
    label_val_path = "VOCdevkit/Mapilary/validation/labels"

    new_label_train_path = "VOCdevkit/Mapilary/training/class21_labels"
    new_label_val_path = "VOCdevkit/Mapilary/validation/class21_labels"

    train_dir = "VOCdevkit/Mapilary/label_train.txt"
    val_dir = "VOCdevkit/Mapilary/label_val.txt"

    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(val_dir,'r').read().splitlines()
    gt_dir          = new_label_val_path

    miou_out_path   = "miou_out_dense_mapi"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')
    blend_dir = os.path.join(miou_out_path, 'blend-results')
    if not os.path.exists(blend_dir):
        os.makedirs(blend_dir)

    # if miou_mode == 0 or miou_mode == 1:
    #     if not os.path.exists(pred_dir):
    #         os.makedirs(pred_dir)
    #
    #     print("Load model.")
    #     deeplab = DeeplabV3(type="dense_mapi")
    #     print("Load model done.")
    #
    #     print("Get predict result.")
    #     for image_id in tqdm(image_ids):
    #         image_path  = os.path.join(image_val_path,image_id+".jpg")
    #         image       = Image.open(image_path)
    #         image       = deeplab.get_miou_png(image)
    #         image.save(os.path.join(pred_dir, image_id + ".png"))
    #     print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        image_ids = {"Type":"mapilary","image_ids":image_ids}
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes, blend_type="dense_mapi")  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)