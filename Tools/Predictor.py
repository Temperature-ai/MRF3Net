#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import os
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Tools.model import model

class predict(object):
    def __init__(self, opt):
        self.model = model(opt)
        # ----------------------------------------------------------------------------------------------------------#
        #   mode用于指定测试的模式：
        #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
        #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
        #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
        #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
        # ----------------------------------------------------------------------------------------------------------#
        self.opt = opt
        self.mode = opt.pre_mode
        self.count = False
        self.name_classes = ["background", "target"]
        self.video_path = opt.video_path
        if self.video_path == '0':
            self.video_path = int(self.video_path)
        self.video_save_path = opt.video_save_path
        self.video_fps = 25.0
        self.dataset_path = opt.dataset_path
        self.dir_save_path = opt.dir_save_path
        self.simplify = opt.simplify
        self.onnx_save_path = opt.onnx_save_path
        self.save_path = 'pre_re/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def pre(self):
        if self.mode == "predict":
            img = self.opt.img
            image = Image.open(img)
            r_image = self.model.detect_image(image, count=self.count, name_classes=self.name_classes)
            r_image = np.array(r_image)
            plt.imshow(r_image)
            plt.imsave(self.save_path + self.opt.img.split('/')[-1], r_image)
            print('Done! Result has been saved to {}'.format(self.save_path + self.opt.img.split('/')[-1]))

        elif self.mode == "video":
            capture = cv2.VideoCapture(self.video_path)
            if self.video_save_path != "":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out = cv2.VideoWriter(self.video_save_path, fourcc, self.video_fps, size)

            ref, frame = capture.read()
            if not ref:
                raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

            fps = 0.0
            while (True):
                t1 = time.time()
                # 读取某一帧
                ref, frame = capture.read()
                if not ref:
                    break
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                # 进行检测
                frame = np.array(self.model.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if self.video_save_path != "":
                    out.write(frame)

                if c == 27:
                    capture.release()
                    break
            print("Video Detection Done!")
            capture.release()
            if self.video_save_path != "":
                print("Save processed video to the path :" + self.video_save_path)
                out.release()
            cv2.destroyAllWindows()

        elif self.mode == "dir_predict":
            import os
            from tqdm import tqdm

            with open(os.path.join(self.dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
                image_ids = f.readlines()
            for image_id in tqdm(image_ids):
                # -------------------------------#
                #   从文件中读取图像
                # -------------------------------#
                image_id = image_id.replace('\n', '')
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
                r_image, save_pr = self.model.detect_image(image)
                if not os.path.exists(self.dir_save_path):
                    os.makedirs(self.dir_save_path)
                save_dir = 'Ours/NUAA-SIRST/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # np.save(save_dir + image_id + '.npy', save_pr)
                r_image.save(os.path.join(self.dir_save_path, image_id + '.png'))
        elif self.mode == "export_onnx":
            self.model.convert_to_onnx(self.simplify, self.onnx_save_path)

        else:
            raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")

