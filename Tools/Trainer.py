import os
import datetime
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.MRF3Net import MRF3Net
from nets.Loss_function import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch

class Train(object):
    def __init__(self, opt):
        
        self.Cuda = opt.Cuda
        self.distributed = opt.Distributed
        self.sync_bn = opt.Sync_bn
        self.fp16 = opt.Fp16
        self.num_classes = opt.num_classes
        self.pretrained = opt.pretrained
        self.model_path = opt.weights
        self.input_shape = [opt.img_size, opt.img_size]

        self.Init_Epoch = 0
        self.UnFreeze_Epoch = opt.Epochs
        self.Unfreeze_batch_size = opt.batch_size

        self.Init_lr = opt.Init_lr
        self.Min_lr = self.Init_lr * 0.01

        self.optimizer_type = opt.optimizer_type
        self.momentum = 0.937
        if self.optimizer_type == 'adam':
            self.weight_decay = 0
        if self.optimizer_type == 'sgd':
            self.weight_decay = 5e-4

        self.lr_decay_type = opt.lr_decay_type
        self.save_period = opt.save_period
        self.save_dir = opt.save_dir

        self.eval_flag = opt.eval_flag
        self.eval_period = opt.eval_period

        self.dataset_path = opt.dataset_path

        self.IoU_loss = opt.IoU_loss
        self.focal_loss = opt.focal_loss

        self.cls_weights = np.ones([self.num_classes], np.float32)
        self.mask_loss_weights = opt.mask_loss_weights
        self.edge_loss_weights = opt.edge_loss_weights

        self.num_workers = opt.num_workers

    def trainer(self):
        ngpus_per_node = torch.cuda.device_count()
        if self.distributed:
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            device = torch.device("cuda", local_rank)
            if local_rank == 0:
                print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
                print("Gpu Device Count : ", ngpus_per_node)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_rank = 0

        # load model
        model = MRF3Net(num_classes=self.num_classes, Train=True).train()
        if not self.pretrained:
            weights_init(model)
        if self.model_path != '' and self.pretrained:
            if local_rank == 0:
                print('Load weights {}.'.format(self.model_path))
            model_dict = model.state_dict()
            pretrained_dict = torch.load(self.model_path, map_location=device)
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
            if local_rank == 0:
                print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
                print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
                print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

        if local_rank == 0:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(self.save_dir, "loss_" + str(time_str))
            loss_history = LossHistory(log_dir, model, input_shape=self.input_shape)
        else:
            loss_history = None

        if self.fp16:
            from torch.cuda.amp import GradScaler as GradScaler
            scaler = GradScaler()
        else:
            scaler = None

        model_train = model.train()

        if self.sync_bn and ngpus_per_node > 1 and self.distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
        elif self.sync_bn:
            print("Sync_bn is not support in one gpu or not distributed.")

        if self.Cuda:
            if self.distributed:
                model_train = model_train.cuda(local_rank)
                model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                        find_unused_parameters=True)
            else:
                model_train = torch.nn.DataParallel(model)
                cudnn.benchmark = True
                model_train = model_train.cuda()

        with open(os.path.join(self.dataset_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
            train_lines = f.readlines()
        with open(os.path.join(self.dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
            val_lines = f.readlines()
        num_train = len(train_lines)
        num_val = len(val_lines)

        if local_rank == 0:
            show_config(
                num_classes=self.num_classes, model_path=self.model_path, dataset_path=self.dataset_path, input_shape=self.input_shape, \
                UnFreeze_Epoch=self.UnFreeze_Epoch, Freeze_batch_size=self.Unfreeze_batch_size, \
                Init_lr=self.Init_lr, Min_lr=self.Min_lr, optimizer_type=self.optimizer_type, momentum=self.momentum,
                lr_decay_type=self.lr_decay_type, \
                save_period=self.save_period, save_dir=self.save_dir, num_workers=self.num_workers, num_train=num_train,
                num_val=num_val
            )

        if True:
            batch_size = self.Unfreeze_batch_size

            nbs = 16
            lr_limit_max = 1e-4 if self.optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if self.optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * self.Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * self.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            optimizer = {
                'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(self.momentum, 0.999), weight_decay=self.weight_decay),
                'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=self.momentum, nesterov=True,
                                 weight_decay=self.weight_decay)
            }[self.optimizer_type]

            lr_scheduler_func = get_lr_scheduler(self.lr_decay_type, Init_lr_fit, Min_lr_fit, self.UnFreeze_Epoch)

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            train_dataset = UnetDataset(train_lines, self.input_shape, self.num_classes, False, self.dataset_path)
            val_dataset = UnetDataset(val_lines, self.input_shape, self.num_classes, False, self.dataset_path)

            if self.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
                batch_size = batch_size // ngpus_per_node
                shuffle = False
            else:
                train_sampler = None
                val_sampler = None
                shuffle = True

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)

            if local_rank == 0:
                eval_callback = EvalCallback(model, self.input_shape, self.num_classes, val_lines, self.dataset_path, log_dir, self.Cuda, \
                                             eval_flag=self.eval_flag, period=self.eval_period)
            else:
                eval_callback = None

            for epoch in range(self.Init_Epoch, self.UnFreeze_Epoch):
                batch_size = self.Unfreeze_batch_size
                nbs = 16
                lr_limit_max = 1e-4 if self.optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if self.optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * self.Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * self.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(self.lr_decay_type, Init_lr_fit, Min_lr_fit, self.UnFreeze_Epoch)

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if self.distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)

                if self.distributed:
                    train_sampler.set_epoch(epoch)

                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

                fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                              epoch_step, epoch_step_val, gen, gen_val, self.UnFreeze_Epoch, self.Cuda, self.IoU_loss, self.focal_loss,
                              self.cls_weights, self.num_classes, self.fp16, scaler, self.save_period, self.save_dir, self.mask_loss_weights, self.edge_loss_weights, local_rank)

                if self.distributed:
                    dist.barrier()

            if local_rank == 0:
                loss_history.writer.close()