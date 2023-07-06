import  numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
epsilon = 1e-5

def ax_adjust(ax):
    '''调整坐标系'''
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
def Get_visual_whiten_img():
    # 获取数据
    X=np.loadtxt('D:\Python_Deeplabv3p\data.txt')

    fig4 = plt.figure(figsize=(8,8))

    # 零均值化
    X_hat= X- np.mean(X)
    ax4 = fig4.add_subplot(221)
    ax_adjust(ax4)
    ax4.scatter(X_hat[0, :40], X_hat[1, :40])
    ax4.scatter(X_hat[0, 41:], X_hat[1, 41:])
    ax4.set_title('零均值化')
    sigma = np.dot(X_hat, X_hat.T) / X_hat.shape[1] # 计算协方差矩阵sigma
    [u, s, v] = np.linalg.svd(sigma) # 计算特征向量矩阵u(对称矩阵的奇异值分解就是特征分解)

    # 旋转数据
    x_rot = np.dot(u.T, X_hat)
    ax4 = fig4.add_subplot(222)
    ax_adjust(ax4)
    ax4.scatter(x_rot[0, :40], x_rot[1, :40])
    ax4.scatter(x_rot[0, 41:], x_rot[1, 41:])
    ax4.set_title('旋 转')

    # PCA白化
    PCA_whitening = np.diag(1. / np.sqrt(s + epsilon)).dot(x_rot)
    ax4 = fig4.add_subplot(223)
    ax_adjust(ax4)
    ax4.scatter(PCA_whitening[0, :40], PCA_whitening[1, :40])
    ax4.scatter(PCA_whitening[0, 41:], PCA_whitening[1, 41:])
    ax4.set_title('PCA whitening')

    # ZCA白化
    ZCA_whitening = np.dot(u,PCA_whitening)
    ax4 = fig4.add_subplot(224)
    ax_adjust(ax4)
    ax4.scatter(ZCA_whitening[0, :40], ZCA_whitening[1, :40])
    ax4.scatter(ZCA_whitening[0, 41:], ZCA_whitening[1, 41:])
    ax4.set_title('ZCA whitening')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
        wspace=None, hspace=0.45)
    plt.savefig("C:/Users/hp/Desktop/毕设/visual whitening pic.png")
    plt.show()


def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion*1.15])

def draw_plot_barh(values, name_classes, plot_title, x_label,save_path, color='royalblue', tick_font_size = 12, plt_show = True):
    #获取当前图表对象
    fig     = plt.gcf()
    axes    = plt.gca()
    #横向的柱状图: plt.bar 竖向; plt.barh 横向
    plt.barh(range(len(values)), values, color=color)
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color=color, va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(save_path)
    if plt_show:
        plt.show()
    plt.close()

def draw_plot_barh_delta(values, delta , name_classes, plot_title, x_label, save_path, color='orange', tick_font_size = 12, plt_show = True):
    #获取当前图表对象
    fig     = plt.gcf()
    axes    = plt.gca()
    #横向的柱状图: plt.bar 竖向; plt.barh 横向
    plt.barh(range(len(values)), values, color=color)
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = "  {}".format(str(val))
        plt.text(val, i, str_val, color=color, va='center', fontweight='bold')
        del_str = " ({:+.2f})".format(delta[i])
        del_spc = 12
        if delta[i] >= 0:
            t = plt.text(val+del_spc, i, del_str, color="red",fontsize=tick_font_size-2, va='center', fontweight='bold')
        else:
            t = plt.text(val+del_spc, i, del_str, color="black",fontsize=tick_font_size-2, va='center', fontweight='bold')

        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(save_path)
    if plt_show:
        plt.show()
    plt.close()

def Get_plot_compared():
    name_classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                    "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle"]
    values=np.array([97.32,79.75,87.62,67.85,63.14,32.05,33.38,49.92,87.55,66.15,91.17,59.16,
                     36.92,88.48,78.53,80.04,53.18,39,73.3,43.12,55.65])

    values2=np.array([97.22,79.48,87.69,71.16,66.08,32.43,35.93,50.54,87.85,70.82,90.8,59.82,
                      41.4,88.2,80.61,82.48,59.18,55.61,74.88,46.14,56.54])
    delta=values2-values

    plot1_title="ASPP Miou=64.92%"
    x1_label="val on Cityscapes Dataset"
    save1_path="C:/Users/hp/Desktop/毕设/aspp_city_miou.png"
    draw_plot_barh(values, name_classes, plot1_title, x1_label,save1_path,color='royalblue')

    plot2_title="Dense+SP Miou=67.37%"
    x2_label="val on Cityscapes Dataset"
    save2_path="C:/Users/hp/Desktop/毕设/dense_city_miou.png"
    draw_plot_barh_delta(values2, delta ,name_classes , plot2_title, x2_label, save2_path,color='orange')

    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]
    values=np.array([93.1 ,82.6 ,41.9 ,83.2 ,62.9 ,70.6 ,93.5 ,85.4 ,
                     88.1 ,35.9 ,79.7 ,51.0 ,80.3 ,78.6 ,81.9 ,80.3,57.1 ,81.9 ,46.6 ,84.7 ,66.3])

    values2=np.array([93.1 ,84.9 ,40.4 ,86.0 ,56.5 ,67.0 ,93.6 ,84.8 ,
                      88.4 ,39.3 ,84.2 ,69.5 ,84.4 ,80.3 ,82.8 ,78.2 ,57.9 ,83.6 ,56.9 ,86.6 ,64.9])
    delta=values2-values
    plot1_title="ASPP Miou=72.63%"
    x1_label="val on PASCAl VOC Dataset"
    save1_path="C:/Users/hp/Desktop/毕设/aspp_voc_miou.png"
    draw_plot_barh(values, name_classes, plot1_title, x1_label,save1_path,color='royalblue')

    plot2_title="Dense+SP Miou=74.44%"
    x2_label="val on PASCAl VOC Dataset"
    save2_path="C:/Users/hp/Desktop/毕设/dense_voc_miou.png"
    draw_plot_barh_delta(values2, delta ,name_classes , plot2_title, x2_label, save2_path,color='orange')

def draw_plot_iwl(values, name_classes, plot_title, x_label, save_path, color='green', tick_font_size = 12, plt_show = True):
    # 获取当前图表对象
    fig = plt.gcf()
    axes = plt.gca()
    # 横向的柱状图: plt.bar 竖向; plt.barh 横向
    plt.barh(range(len(values)), values, color=color)
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color=color, va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(save_path)
    if plt_show:
        plt.show()
    plt.close()

def Get_plot_compared_plus():

    name_classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                    "truck", "bus", "train", "motorcycle", "bicycle"]

    aspp_mapi = np.array([80.16,34.71,73.33,20.41,29.82,26.15,21.09,25.35,78.76,43.57,90.4,33.04,31.14,74.52,29.9,26.51,16.71,16.69,26.08])
    desne_mapi = np.array([83.77,31.54,71.15,18.57,27.04,24.97,22.04,27.87,79.12,42.58,92.15,37.21,30.42,72.53,20.71,29.71,13.51,26.69,26.02])
    iwl_mapi = np.array([86.84,36.07,70.83,26.7,40.12,39.17,36.9,55.74,85.15,45.59,93.32,63.67,33.15,83.43,29.4,27.27,17.66,35.04,44.78])

    iwl_city = np.array([97.42,79.32,89.47,41.39,51.53,51.82,54.98,69.04,90.86,53.6,92.04,73.65,50.07,91.89,59.84,69.23,43.21,49.59,69.58])

    delta1 = desne_mapi - aspp_mapi
    delta2 = iwl_mapi - aspp_mapi

    plot1_title="ASPP Miou=40.97%"
    x1_label="val on Mapillary Dataset"
    save1_path="C:/Users/hp/Desktop/毕设/aspp_mapi_miou.png"
    draw_plot_barh(aspp_mapi, name_classes, plot1_title, x1_label, save1_path, color='royalblue')

    plot2_title = "Dense+SP Miou=40.92%"
    x2_label = "val on Mapillary Dataset"
    save2_path = "C:/Users/hp/Desktop/毕设/dense_mapi_miou.png"
    draw_plot_barh_delta(desne_mapi, delta1, name_classes, plot2_title, x2_label, save2_path, color='orange')

    plot3_title = "Dense+SP+IWL Miou=50.04%"
    x3_label = "val on Mapillary Dataset"
    save3_path = "C:/Users/hp/Desktop/毕设/iwl_mapi_miou.png"
    draw_plot_barh_delta(iwl_mapi, delta2, name_classes, plot3_title, x3_label, save3_path, color='green')



# Get_plot_compared()
Get_plot_compared_plus()