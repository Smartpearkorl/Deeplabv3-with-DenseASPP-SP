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

# --------------------------------------------#
#   运行前修改 三个地方: 一名字，两路径
#   deeplab.py : line 31 参数模型路径 line 116 模型类型
#   utils_metrics : line 88 lend_image 路径
# --------------------------------------------#

if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 21
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                    "traffic sign","vegetation","terrain", "sky", "person", "rider", "car",
                    "truck", "bus", "caravan" , "trailer", "train","motorcycle", "bicycle"]
    # -------------------------------------------------------#
    #  一般测量的是val的miou,但还是留了train的目录
    #   流程 : 读取原图的txt获得样本名字的列表 -> 经过网路 ->获得pred的图
    #   ->读取gt的txt获得样本名字的列表 —> 和pred的图计算miou
    # -------------------------------------------------------#
    image_train_dir = "VOCdevkit/Cityscapes/image_train.txt"
    image_val_dir = "VOCdevkit/Cityscapes/image_val.txt"
    label_train_dir = "VOCdevkit/Cityscapes/label_train.txt"
    label_val_dir = "VOCdevkit/Cityscapes/label_val.txt"

    image_train_path = "VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/train"
    image_val_path = "VOCdevkit/Cityscapes/Cityspaces_leftImg8bit/val"
    label_train_path = "VOCdevkit/Cityscapes/gtFine/train"
    label_val_path = "VOCdevkit/Cityscapes/gtFine/val"

    # 获取原图的样本名字
    image_ids = open(image_val_dir, 'r').read().splitlines()
    #image_ids = open(image_train_dir, 'r').read().splitlines()
    # -------------------------------------------------------#
    #   gt_dir 和 image_ids搭配
    #   如果image_ids是image_train_dir，则gt_dir=label_train_path;反之则为label_val_path
    # -------------------------------------------------------#
    gt_dir = label_val_path
    #gt_dir = label_train_path

    miou_out_path = "miou_out_dense_city"
    pred_dir = os.path.join(miou_out_path, 'detection-results')
    blend_dir = os.path.join(miou_out_path, 'blend-results')
    if not os.path.exists(blend_dir):
        os.makedirs(blend_dir)

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        if not os.path.exists(blend_dir):
            os.makedirs(blend_dir)

        print("Load model.")
        deeplab = DeeplabV3(type="dense_city")
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(image_val_path , image_id )
            # image_path = os.path.join(image_train_path, image_id)
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            # ------------------------------#
            # image_id = frankfurt/frankfurt_000000_000294_leftImg8bit.png
            # 需要将 frankfurt/截掉
            # ------------------------------#
            temp_lines = image_id.split('/')[1]
            image.save(os.path.join(pred_dir, temp_lines))
        print("Get predict result done.")


    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        # -------------------------------------------------------#
        #   gt_dir : "VOCdevkit/Cityscapes/gtFine/val" or "VOCdevkit/Cityscapes/gtFine/train"
        #   image_ids :  label的名字和predict的名字不同，所以传入字典 :
        #   image_ids = {'Type':1,'image_lines':image_val_lines,'label_lines':label_val_lines} 或者
        #               {'Type':0,'image_lines':image_train_lines,'label_lines':label_train_lines}
        # -------------------------------------------------------#

        # 验证集
        with open(image_val_dir, 'r') as f:
            image_val_lines = f.readlines()
        with open(label_val_dir, 'r') as f:
            label_val_lines = f.readlines()
        # with open(image_train_dir, 'r') as f:
        #     image_val_lines = f.readlines()
        # with open(label_train_dir, 'r') as f:
        #     label_val_lines = f.readlines()
        image_ids = {'Type':1,'image_lines':image_val_lines,'label_lines':label_val_lines}

        #训练集
        # with open(image_train_dir, 'r') as f:
        #     image_train_lines = f.readlines()
        # with open(label_train_dir, 'r') as f:
        #     label_train_lines = f.readlines()
        # image_ids = {'Type':0,'image_lines':image_train_lines,'label_lines':label_train_lines}

        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes, blend_type="dense_city")  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)