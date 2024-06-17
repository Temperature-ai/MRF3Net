import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from Tools.model import model
from utils.utils_metrics import compute_mIoU, show_results, PD_FA, mIoU, nIoU

class validation(object):
    def __init__(self, opt):
        self.opt = opt
        self.num_classes = opt.num_classes
        self.name_classes = ["background", "target"]
        self.dataset_path = opt.dataset_path
        self.miou_out_path = opt.val_out_path
        self.PD_FA = PD_FA()
        self.mIoU = mIoU()
        self.nIoU = nIoU()

    def val(self):
        image_ids = open(os.path.join(self.dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
        gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")

        pred_dir = os.path.join(self.miou_out_path, 'detection-results')

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        Model = model(self.opt)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):

            try:
                image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
            except FileNotFoundError:
                try:
                    image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".png")
                    image = Image.open(image_path)
                except FileNotFoundError:
                    image_path = os.path.join(self.dataset_path,
                                              "VOC2007/JPEGImages/" + image_id.replace('_pixels0', '') + ".png")
                    image = Image.open(image_path)

            image = Model.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, self.num_classes, self.PD_FA, self.mIoU, self.nIoU, self.name_classes)
        print("Get miou done.")
        # show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
    
    def val_nIoU(self):
        image_ids = open(os.path.join(self.dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
        gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")

        pred_dir = os.path.join(self.miou_out_path, 'detection-results')

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        Model = model(self.opt)
        print("Load model done.")

        print("Get predict result.")
        total_IoU = []
        for image_id in tqdm(image_ids):

            try:
                image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
            except FileNotFoundError:
                try:
                    image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages/" + image_id + ".png")
                    image = Image.open(image_path)
                except FileNotFoundError:
                    image_path = os.path.join(self.dataset_path,
                                              "VOC2007/JPEGImages/" + image_id.replace('_pixels0', '') + ".png")
                    image = Image.open(image_path)

            image = Model.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))

            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, [image_id], self.num_classes, self.PD_FA, self.mIoU, self.name_classes)
            total_IoU.append(IoUs)
        print("Get miou done.")
        print("nIoU: {:.3f} %".format(np.mean(total_IoU) * 100))
        # show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)