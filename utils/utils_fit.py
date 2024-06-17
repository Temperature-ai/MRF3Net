import os

import torch
from nets.Loss_function import CE_Loss, Focal_Loss, IoU_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, IoU_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, mask_loss_weights, edge_loss_weights, local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    re_loss         = 0
    val_loss        = 0
    val_f_score     = 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels, edges = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                edges   = edges.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs[0], pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs[0], pngs, weights, num_classes = num_classes)

            if IoU_loss:
                main_dice = IoU_Loss(outputs[0], pngs)
                loss      = loss + main_dice

            mask_loss  = 0
            for out in outputs[1]:
                mask_loss +=  (IoU_Loss(out, pngs) * mask_loss_weights)
            loss += mask_loss

            edge_loss = 0
            for out in outputs[2]:
                edge_loss += (CE_Loss(out, edges, weights, num_classes = num_classes)) * edge_loss_weights
            loss += edge_loss

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs[0], labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs[0], pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs[0], pngs, weights, num_classes = num_classes)

                if IoU_loss:
                    main_dice = IoU_Loss(outputs[0], pngs)
                    loss      = loss + main_dice

                mask_loss = 0
                for out in outputs[1]:
                    mask_loss += (IoU_Loss(out, pngs) * mask_loss_weights)
                loss += mask_loss

                edge_loss = 0
                for out in outputs[2]:
                    edge_loss += (CE_Loss(out, edges, weights, num_classes=num_classes)) * edge_loss_weights
                loss += edge_loss

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs[0], labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        re_loss         += (mask_loss.item() + edge_loss.item() )
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'mid_loss'  : re_loss / (iteration + 1),
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels, edges = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                edges = edges.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs[0], pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs[0], pngs, weights, num_classes = num_classes)

            if IoU_loss:
                main_dice = IoU_Loss(outputs[0], pngs)
                loss = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs[0], labels)
            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('F1_SCORE: %.3f ' % (val_f_score / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))

def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, IoU_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, mask_loss_weights, edge_loss_weights, local_rank=0):
    total_loss = 0
    total_f_score = 0
    re_loss = 0
    val_loss = 0
    val_f_score = 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels, edges = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                edges = edges.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs[0], pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs[0], pngs, weights, num_classes=num_classes)

            if IoU_loss:
                main_dice = IoU_Loss(outputs[0], pngs)
                loss = loss + main_dice

            mask_loss = 0
            for out in outputs[1]:
                mask_loss += (IoU_Loss(out, pngs) * mask_loss_weights)
            loss += mask_loss

            edge_loss = 0
            for out in outputs[2]:
                edge_loss += (CE_Loss(out, edges, weights, num_classes=num_classes)) * edge_loss_weights
            loss += edge_loss
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if IoU_loss:
                    main_dice = IoU_loss(outputs, labels)
                    loss      = loss + main_dice

                mask_loss = 0
                for out in outputs[1]:
                    mask_loss += (IoU_Loss(out, pngs) * 0.5)
                loss += mask_loss

                edge_loss = 0
                for out in outputs[2]:
                    edge_loss += (CE_Loss(out, edges, weights, num_classes=num_classes))
                loss += edge_loss

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()
        re_loss += (mask_loss.item() + edge_loss.item())
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'mid_loss': re_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))