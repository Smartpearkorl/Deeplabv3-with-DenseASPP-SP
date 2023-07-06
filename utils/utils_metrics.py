import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


#--------------------------------------------#
#   f-score = precision * recall * 2 / (precision + recall)
#   precision = tp / (tp + fp )
#   recall =  tp / (tp + fn )
# ->f-score =  2*tp / ( 2*tp + fp + fn )
# ->f-score =  ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
# --------------------------------------------#
def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    # ----------------------#
    #   inputs: torch.tensor(shape=(n=batch_size,c=class_num - 1,w=512,h=512))
    #   target: torch.tensor(n=batch_size,shape=(w=512,h=512,c=class_num))
    # ----------------------#
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # ----------------------#
    #   temp_inputs: n,c,h,w --> n,h*w,c 并对通道求softmax
    #   temp_target: n,h,w,c --> n,h*w,c 并对通道求softmax
    # ----------------------#
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    # torch.gt(a,b) a中元素严格大于b,则返回1；否则返回0。——>输出类似 target（label）的 one-hot 值
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)



colors = [ (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153),
           (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60),
           (255,  0,  0), (  0,  0,142), (  0,  0, 70), (  0, 60,100), (  0,  0, 90), (  0,  0,110),
           (  0, 80,100), (  0,  0,230), (119, 11, 32), (128 , 128 , 128)]



# -------------------------------------------------------------------#
# Deeplab path
# -------------------------------------------------------------------#
# blend_save_dir="miou_out_cityscapes/blend-results"

# -------------------------------------------------------------------#
# Deeplab_Dense path
# -------------------------------------------------------------------#,
blend_save_dir="miou_out_Dense_City/blend-results"

aspp_voc_blend_save_path = "miou_out_aspp_voc/blend-results"
aspp_city_blend_save_path = "miou_out_aspp_city/blend-results"
aspp_mapi_blend_save_path = "miou_out_aspp_mapi/blend-results"
dense_voc_blend_save_path = "miou_out_dense_voc/blend-results"
dense_city_blend_save_path = "miou_out_dense_city/blend-results"
dense_mapi_blend_save_path = "miou_out_dense_mapi/blend-results"
def orig_lend_pred(orig_imgs,pred_imgs,blend_type):
    save_num = 20
    save_path = aspp_voc_blend_save_path
    # -------------------------------------------------------------------#
    # mapi数据集
    # -------------------------------------------------------------------#
    if blend_type=="aspp_mapi" or blend_type=="dense_mapi":
        if blend_type=="aspp_mapi":
            save_path = aspp_mapi_blend_save_path
        elif blend_type=="dense_mapi":
            save_path = dense_mapi_blend_save_path
        assert len(orig_imgs) == len(pred_imgs)
        print("Get Blend-image num : %d  path : %s  " % (save_num, save_path))
        inds = np.random.randint(0, len(orig_imgs), size=save_num)
        for ind in range(save_num):
            ind = int(inds[ind])
            orig_img = Image.open(orig_imgs[ind])
            pred = Image.open(pred_imgs[ind])

            orininal_h = np.array(orig_img).shape[0]
            orininal_w = np.array(orig_img).shape[1]

            color_np = np.array(colors, np.uint8)
            np_pred = np.reshape(np.array(pred), [-1])
            pre_img = np.reshape(color_np[np_pred], [orininal_h, orininal_w, -1])
            pre_img = Image.fromarray(np.uint8(pre_img))

            blend = Image.blend(orig_img, pre_img, 0.5)
            temp_lines = orig_imgs[ind].split('/')[-1]
            blend.save(os.path.join(save_path, temp_lines))
        print("Get Blend-image Finished")
    # -------------------------------------------------------------------#
    # VOC数据集
    # -------------------------------------------------------------------#
    elif blend_type=="aspp_voc" or blend_type=="dense_voc":
        if blend_type=="aspp_voc":
            save_path = aspp_voc_blend_save_path
        elif blend_type=="dense_voc":
            save_path = dense_voc_blend_save_path
        assert len(orig_imgs) == len(pred_imgs)
        print("Get Blend-image num : %d  path : %s  " % (save_num, save_path))
        inds = np.random.randint(0, len(orig_imgs), size=save_num)
        for ind in range(save_num):
            ind = int(inds[ind])
            orig_img = Image.open(orig_imgs[ind])
            pred = Image.open(pred_imgs[ind])

            orininal_h = np.array(orig_img).shape[0]
            orininal_w = np.array(orig_img).shape[1]

            color_np = np.array(colors, np.uint8)
            np_pred = np.reshape(np.array(pred), [-1])
            pre_img = np.reshape(color_np[np_pred], [orininal_h, orininal_w, -1])
            pre_img = Image.fromarray(np.uint8(pre_img))

            blend = Image.blend(orig_img, pre_img, 0.5)
            temp_lines = orig_imgs[ind].split('/')[-1]
            blend.save(os.path.join(save_path, temp_lines))
        print("Get Blend-image Finished")

    # -------------------------------------------------------------------#
    # cityscapes数据集
    # -------------------------------------------------------------------#
    elif blend_type=="aspp_city" or blend_type=="dense_city":
        if blend_type=="aspp_city":
            save_path = aspp_city_blend_save_path
        elif blend_type=="dense_city":
            save_path = dense_city_blend_save_path
        assert len(orig_imgs) == len(pred_imgs)
        print("Get Blend-image num : %d  path : %s  " % (save_num, save_path))
        inds = np.random.randint(0, len(orig_imgs), size=save_num)
        for ind in range(save_num):
            ind = int(inds[ind])
            orig_img = Image.open(orig_imgs[ind])
            pred = Image.open(pred_imgs[ind])

            color_np = np.array(colors, np.uint8)
            np_pred = np.reshape(np.array(pred), [-1])
            pre_img = np.reshape(color_np[np_pred], [1024, 2048, -1])
            pre_img = Image.fromarray(np.uint8(pre_img))

            blend = Image.blend(orig_img, pre_img, 0.5)
            temp_lines = orig_imgs[ind].split('/')[-1].replace("leftImg8bit", "blendImg")
            blend.save(os.path.join(save_path, temp_lines))

        print("Get Blend-image Finished")



def City_orig_blend_pred(orig_imgs,pred_imgs,blend_type):
    # for ind in range(len(orig_imgs)):
    assert len(orig_imgs)==len(pred_imgs)
    save_num=20
    print("Get Blend-image num : %d  path : %s  "%(save_num,blend_save_dir))
    inds = np.random.randint(0, len(orig_imgs) ,size=save_num)
    for ind in range(save_num):
        ind=int(inds[ind])
        orig_img = Image.open(orig_imgs[ind])
        pred = Image.open(pred_imgs[ind])

        color_np = np.array(colors, np.uint8)
        np_pred = np.reshape(np.array(pred), [-1])
        pre_img = np.reshape(color_np[np_pred], [1024, 2048, -1])
        pre_img = Image.fromarray(np.uint8(pre_img))

        blend = Image.blend(orig_img, pre_img, 0.5)
        temp_lines = orig_imgs[ind].split('/')[-1].replace("leftImg8bit","blendImg")
        blend.save(os.path.join(blend_save_dir, temp_lines))

    print("Get Blend-image Finished" )

# -------------------------------------------------------------------#
# VOC path
# -------------------------------------------------------------------#
VOC_blend_save_dir="miou_out/blend-results"
def VOC_orig_blend_pred(orig_imgs,pred_imgs,blend_type):
    # for ind in range(len(orig_imgs)):
    assert len(orig_imgs)==len(pred_imgs)
    save_num=20
    print("Get Blend-image num : %d  path : %s  "%(save_num,VOC_blend_save_dir))
    inds = np.random.randint(0, len(orig_imgs) ,size=save_num)
    for ind in range(save_num):
        ind=int(inds[ind])
        orig_img = Image.open(orig_imgs[ind])
        pred = Image.open(pred_imgs[ind])

        orininal_h  = np.array(orig_img).shape[0]
        orininal_w  = np.array(orig_img).shape[1]

        color_np = np.array(colors, np.uint8)
        np_pred = np.reshape(np.array(pred), [-1])
        pre_img = np.reshape(color_np[np_pred], [orininal_h, orininal_w, -1])
        pre_img = Image.fromarray(np.uint8(pre_img))

        blend = Image.blend(orig_img, pre_img, 0.5)
        temp_lines = orig_imgs[ind].split('/')[-1]
        blend.save(os.path.join(VOC_blend_save_dir, temp_lines))

    print("Get Blend-image Finished" )

def compute_mIoU(gt_dir, pred_dir, input_ids_lines, num_classes, name_classes=None , blend_type="aspp_voc"):
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    # check_label = np.zeros(num_classes+1)
    # check_predict = np.zeros(num_classes+1)
    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #   Cityscapes :
    #   gt_dir = 'VOCdevkit/Cityscapes/gtFine/train' or 'VOCdevkit/Cityscapes/gtFine/val'
    #   pred_dir = '.temp_miou_out/detection-results'
    #   VOC :
    #   gt_dir = 'VOCdevkit/VOC2007/SegmentationClass '
    #   pred_dir = '.temp_miou_out/detection-results'
    #------------------------------------------------#
    # -------------------------------#
    #   mapilary数据集 :
    # -------------------------------#
    if type(input_ids_lines) == dict and input_ids_lines['Type']=='mapilary':
        print("mapilary数据集")
        image_ids_lines = [image_id.split()[0] for image_id in input_ids_lines["image_ids"]]
        gt_imgs = [join(gt_dir, x + ".png") for x in image_ids_lines]
        pred_imgs = [join(pred_dir, x + ".png") for x in image_ids_lines]

    # -------------------------------#
    #   Cityscapes数据集 :label的名字和predict的名字不同，所以传入字典
    # -------------------------------#
    elif type(input_ids_lines)==dict:
        print("Cityscapes数据集")
        # 不区分训练集还是验证集，因为 gt_dir = 'VOCdevkit/Cityscapes/gtFine/train' or 'VOCdevkit/Cityscapes/gtFine/val'
        #标签样本地址
        label_lines = [image_id.split()[0] for image_id in input_ids_lines['label_lines']]
        gt_imgs     = [join(gt_dir, x) for x in label_lines]
        #预测样本地址
        # image_id = frankfurt/frankfurt_000000_000294_leftImg8bit.png\n
        # 需要将 frankfurt/截掉
        # ------------------------------#
        temp_lines = [image_id.split('/')[1] for image_id in input_ids_lines['image_lines']]
        pred_lines = [image_id.split()[0] for image_id in temp_lines]
        pred_imgs   = [join(pred_dir, x) for x in pred_lines]

    # -------------------------------#
    #   VOC数据集 : label的名字和predict的名字相同，所以传入列表
    # -------------------------------#
    else:
        print("VOC数据集")
        image_ids_lines = [image_id.split()[0] for image_id in input_ids_lines]
        gt_imgs     = [join(gt_dir, x + ".png") for x in image_ids_lines]
        pred_imgs   = [join(pred_dir, x + ".png") for x in image_ids_lines]

    # ------------------------------------------------#
    #   是否将预测图与原图blend
    # ------------------------------------------------#
    blend_flag = False
    if blend_flag and type(input_ids_lines)==dict and input_ids_lines['Type']=='mapilary':
        # 获得原图地址
        image_ids_lines = [image_id.split()[0] for image_id in input_ids_lines["image_ids"]]
        orig_imgs = [("VOCdevkit/Mapilary/validation/images/" + x + ".jpg") for x in image_ids_lines]
        orig_lend_pred(orig_imgs, pred_imgs, blend_type)

    elif blend_flag and type(input_ids_lines)==dict and (input_ids_lines['Type']==0 or input_ids_lines['Type']==1):
        #获得原图地址
        orig_imgs=[ x.replace("gtFine_labelTrainIds","leftImg8bit").replace("gtFine","Cityspaces_leftImg8bit") for x in gt_imgs]
        orig_lend_pred(orig_imgs,pred_imgs,blend_type)
    elif blend_flag:
        #获得原图地址
        image_ids_lines = [image_id.split()[0] for image_id in input_ids_lines]
        orig_imgs =  [( "VOCdevkit/VOC2007/JPEGImages/"+ x + ".jpg") for x in image_ids_lines]
        orig_lend_pred(orig_imgs, pred_imgs, blend_type)


    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)):
    #for ind in range(100):
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))
        # temp = gt_imgs[ind]
        # if "frankfurt_000001_046272" in gt_imgs[ind]:
        #     print("---")
        label[label>=num_classes] = num_classes

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        # hist += fast_hist(label.flatten(), label.flatten(), num_classes)
        # check_label +=np.bincount(label.flatten().astype(int) , minlength=num_classes+1)
        # check_predict +=np.bincount(pred.flatten().astype(int) , minlength=num_classes+1)

        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mRecall-{:0.2f}%; mPrecision-{:0.2f}%; Accuracy-{:0.2f}%;'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * np.nanmean(per_class_Precision(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    Accuracy = per_Accuracy(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #------------------------------------------------#
    if name_classes is not None:
        fmt = '%-30s %-15s %-15s %-15s'

        for ind_class in range(num_classes):
            print(fmt % ('===>' + name_classes[ind_class],
                         ':Iou ' + str(round(IoUs[ind_class] * 100, 2)),
                         ':Recall ' + str(round(PA_Recall[ind_class] * 100, 2)),
                         ':Precision ' + str(round(Precision[ind_class] * 100, 2))))

            # print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
            #     + '; Recall -' + str(round(PA_Recall[ind_class] * 100, 2))
            #     + '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2))
          + ';  mRecall: ' + str(round(np.nanmean(PA_Recall) * 100, 2))
          + ';  mPrecision: ' + str(round(np.nanmean(Precision) * 100, 2))
          + ';  Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))

    return np.array(hist, np.int), IoUs, PA_Recall, Precision

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    #获取当前图表对象
    fig     = plt.gcf()
    axes    = plt.gca()
    #横向的柱状图: plt.bar 竖向; plt.barh 横向
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()



def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))


    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)

        IoUs = per_class_iu(hist)
        PA_Recall = per_class_PA_Recall(hist)
        Precision = per_class_Precision(hist)
        Accuracy = per_Accuracy(hist)

        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])

        writer_list.append([''])
        writer_list.append([' ']+["mean"] + [str(c) for c in name_classes])
        # Ious
        writer_list.append(["Iou"] + [str(round(np.nanmean(IoUs)*100, 2))] +  [str(round(x * 100, 2)) for x in IoUs ])
        # PA_Recall
        writer_list.append(["Recall"] + [str(round(np.nanmean(PA_Recall)*100, 2))] + [str(round(x * 100, 2)) for x in PA_Recall])
        # Precision
        writer_list.append(["Precision"] + [str(round(Accuracy*100, 2))] + [str(round(x * 100, 2)) for x in Precision])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))


    # with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
    #     writer          = csv.writer(f)
    #     writer_list     = []
    #     writer_list.append([' '] + [str(c) for c in name_classes])
    #     for i in range(len(hist)):
    #         writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
    #     writer.writerows(writer_list)
    # print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))




