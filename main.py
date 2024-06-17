import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import argparse
from Tools.Trainer import Train
from Tools.Validator import validation
from Tools.Predictor import predict
from utils.caulate_FLOPs import show_model_params

parser = argparse.ArgumentParser(description="Main function")
parser.add_argument("--Task", default='prediction', type=str, help="Select from train, prediction and validation")
parser.add_argument("--device", default="0", type=str, help="Device number")
# Task: Train
parser.add_argument("--pretrained", default=True, type=bool, help="Whether load pretrained weights")
parser.add_argument("--weights", default='weights/IRSTD.pth', type=str, help="Pretrained weights path")
parser.add_argument("--img_size", default=512, type=int, help="Image size for training")
parser.add_argument("--num_classes", default=2, type=int, help="The number of class in the dateset")
parser.add_argument("--Epochs", default=400, type=int, help="Total epochs of training")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size of training")
parser.add_argument("--Init_lr", default=1e-3, type=float, help="Batch size of training")
parser.add_argument("--optimizer_type", default='adam', type=str, help="Optimizer, adam or sgd")
parser.add_argument("--lr_decay_type", default='cos', type=str, help="cos or step")
parser.add_argument("--save_period", default=600, type=int)
parser.add_argument("--save_dir", default='logs/', type=str, help='Model save path of experiment')
parser.add_argument("--eval_flag", default=True, type=bool, help='Whether evaluate the model during training')
parser.add_argument("--eval_period", default=10, type=int, help='Evaluation period')
parser.add_argument("--dataset_path", default='/ssd/home/zxh/paper_work/IRSTD/IRSTD_paper_1/IRSTD_data/datasets/NUDT-SIRST/', type=str, help='Dataset path')
parser.add_argument("--IoU_loss", default=True, type=bool, help='Whether use IoU_loss for training')
parser.add_argument("--focal_loss", default=False, type=bool, help='Whether use focal_loss for training')
parser.add_argument("--mask_loss_weights", default=0.01, type=float, help='mask_loss_weights')
parser.add_argument("--edge_loss_weights", default=0.01, type=float, help='edge_loss_weights')
parser.add_argument("--Cuda", default=True, type=bool, help="Whether use cuda to train the model")
parser.add_argument("--Distributed", default=False, type=bool)
parser.add_argument("--Sync_bn", default=False, type=bool, help="Sync_batch")
parser.add_argument("--Fp16", default=False, type=bool, help="Whether use Fp14 mode to train the model")
parser.add_argument("--num_workers", default=4, type=int, help='How many works used in the training')

# Task: val
parser.add_argument("--val_out_path", default='val_results/NUDT', type=str, help="Save path for validation results")

# Task: pre
parser.add_argument("--pre_mode", default='predict', type=str, help="Select from predict, video, dir_predict adn export_onnx")
parser.add_argument("--img", default='/ssd/home/zxh/paper_work/IRSTD/IRSTD_paper_1/IRSTD_data/datasets/IRSTD-1K/VOC2007/JPEGImages/XDU189.jpg', type=str, help="img for prediction")
parser.add_argument("--video_path", default='0', type=str, help="Video_path for prediction")
parser.add_argument("--video_save_path", default='results/result.mp4', type=str, help="Result video save path")
parser.add_argument("--dir_save_path", default='results/Ours/IRSTD-1K', type=str, help="Save Path of prediction results ")
parser.add_argument("--simplify", default=True, type=bool, help="Whether simplify model when export")
parser.add_argument("--onnx_save_path", default='onnx_model/', type=str, help="Save Path of onnx model ")

opt = parser.parse_args()

if __name__ == "__main__":

    if opt.Task == 'train':
        show_model_params(opt.num_classes, False)
        Train(opt).trainer()

    if opt.Task == 'validation':
        validation(opt).val()
        # validation(opt).val_nIoU()

    if opt.Task == 'prediction':
        predict(opt).pre()
