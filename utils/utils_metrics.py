import csv
import os
from os.path import join
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure
from tqdm import tqdm

class nIoU():

    def __init__(self):
        super(nIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):

        correct, labeled = batch_pix_accuracy(preds, labels)

        inter, union = batch_intersection_union(preds, labels)
        self.total_correct = correct
        self.total_label = labeled
        self.total_inter = inter
        self.total_union = union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        return float(pixAcc), IoU.mean()

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        
class mIoU():

    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class PD_FA():
    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array(preds).astype('int64')
        labelss = np.array(labels).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 1:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel += np.sum(self.dismatch)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, Final_FA


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.astype('float64'), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).astype('int64')
    pixel_labeled = (target > 0).astype('int64').sum()
    pixel_correct = (((predict == target).astype('int64')) * ((target > 0)).astype('int64')).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).astype('int64')
    if len(target.shape) == 3:
        target = np.expand_dims(target.astype('int64'), axis=1)
    elif len(target.shape) == 4:
        target = target.astype('int64')
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).astype('int64'))

    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice系数
    # --------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


# 设标签宽W，长H
def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_class_Fa(hist):
    loc = [hist[1, 0], hist[0, 1]]
    return loc / np.sum(hist)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)



def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, PD_FA, mIoU, nIoU, name_classes=None):
    # size = (512, 512)
    # print('Num classes', num_classes)
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    # ------------------------------------------------#
    if gt_dir.find('NUAA')!=-1 :
        gt_imgs = [join(gt_dir, x + "_pixels0.png") for x in png_name_list]
    else:
        gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    count_label_target = 0
    count_pred_target = 0
    total_nIoU = []
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        # ------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))
        pred_size = (pred.shape[0], pred.shape[1])
        # ------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        # ------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))
        label_size = (label.shape[0], label.shape[1])
        if np.max(label) > 1:
            label[label > 0] = 1
        th  = 0.5
        
        try:
            PD_FA.update(pred > th, label, label_size)
            mIoU.update(np.expand_dims(np.expand_dims(pred, 0), 0) > th, np.expand_dims(label, 0))
            nIoU.reset()
            nIoU.update(np.expand_dims(np.expand_dims(pred, 0), 0) > th, np.expand_dims(label, 0))
            total_nIoU.append(nIoU.get()[1])
        except:
            if pred_size!=label_size:
                label = cv2.resize(label, (pred_size[1], pred_size[0]), interpolation=cv2.INTER_NEAREST )
            PD_FA.update(pred > th, label, label_size)
            mIoU.update(np.expand_dims(np.expand_dims(pred, 0), 0) > th, np.expand_dims(label, 0))
            nIoU.reset()
            nIoU.update(np.expand_dims(np.expand_dims(pred, 0), 0) > th, np.expand_dims(label, 0))
            total_nIoU.append(nIoU.get()[1])
            
        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    _, mean_IOU = mIoU.get()
    IoUs = mean_IOU
    PD, FA = PD_FA.get()

    print('===> mIoU: ' + str(round(IoUs * 100, 3)) + '% ; Fa: ' + str(round(FA * 1e6, 3)) + '*1e-6'
          + '; Pd: ' + str(round(PD * 100, 3)) + '; nIoU: ' + str(round(np.mean(total_nIoU) * 100, 3))  + '%')
    return np.array(hist, np.int64), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
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
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100), "Precision", \
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
