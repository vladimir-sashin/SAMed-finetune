import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from datasets.yolo_dataset import to_samed_format
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic

# import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_model(model, snapshot_path, model_name):
    save_mode_path = os.path.join(snapshot_path, f'{model_name}' + '.pth')
    try:
        model.save_lora_parameters(save_mode_path)
    except:  # FIXME: specify exception
        model.module.save_lora_parameters(save_mode_path)
    return save_mode_path


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def worker_init_fn(worker_id):
    random.seed(torch.initial_seed() + worker_id)


def trainer_yolo(args, yolo_cfg, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    # max_iterations = args.max_iterations

    yolo_cfg.imgsz = args.img_size
    data = check_det_dataset(args.dataset_path)
    db_train = build_yolo_dataset(
        yolo_cfg,
        img_path=data["train"],
        batch=batch_size,
        data=data,
        mode="train",
    )
    db_train.transforms.append(to_samed_format)
    db_test = build_yolo_dataset(
        yolo_cfg,
        img_path=data["val"],
        batch=batch_size,
        data=data,
        mode="val",
    )
    db_test.transforms.append(to_samed_format)
    train_set_size = len(db_train)
    print("The length of train set is: {}".format(train_set_size))
    val_set_size = len(db_test)
    print("The length of val set is: {}".format(val_set_size))

    trainloader = build_dataloader(
        db_train, batch_size, yolo_cfg.workers, shuffle=True, rank=-1
    )
    test_loader = build_dataloader(
        db_test, batch_size, yolo_cfg.workers, shuffle=False, rank=-1
    )

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    iters_per_epoch = len(trainloader)
    max_iterations = args.max_epochs * iters_per_epoch  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_val_loss = 1e10
    best_val_epoch = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_dice = 0.0
        running_lr = 0.0

        if epoch_num == (max_epoch - yolo_cfg.close_mosaic):
            logging.info('Closing dataloader mosaic')
            if hasattr(trainloader.dataset, 'mosaic'):
                trainloader.dataset.mosaic = False
            if hasattr(trainloader.dataset, 'close_mosaic'):
                trainloader.dataset.close_mosaic(hyp=yolo_cfg)
            trainloader.reset()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, low_res_label_batch = sampled_batch['img'], sampled_batch['masks']  # [b, c, h, w], [b, h, w]
            # Draw debug images:
            # img = image_batch[1]
            # cv2.imshow("image", img.permute(1, 2, 0).numpy())
            # cv2.waitKey(0)
            # label_img = low_res_label_batch[1].numpy().astype("int8") * int(255 / num_classes)
            # cv2.imshow("down-sampled mask", label_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            image_batch, low_res_label_batch = image_batch.to(DEVICE), low_res_label_batch.to(DEVICE)
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('train_iter/lr', lr_, iter_num)
            writer.add_scalar('train_iter/total_loss', loss, iter_num)
            writer.add_scalar('train_iter/loss_ce', loss_ce, iter_num)
            writer.add_scalar('train_iter/loss_dice', loss_dice, iter_num)

            running_loss += loss.item() * batch_size
            running_loss_ce += loss_ce.item() * batch_size
            running_loss_dice += loss_dice.item() * batch_size

            running_lr += lr_

            logging.info('iteration %d, loss: %f, loss_ce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                labs = low_res_label_batch[1, ...].unsqueeze(0) * int(255 / num_classes)
                writer.add_image('train/GroundTruth', labs, iter_num)

        # Epoch losses and lr
        epoch_loss = running_loss / train_set_size
        epoch_loss_ce = running_loss_ce / train_set_size
        epoch_loss_dice = running_loss_dice / train_set_size
        epoch_lr = running_lr / iters_per_epoch
        # Track epoch losses and lr
        writer.add_scalar('train_epoch/lr', epoch_lr, epoch_num)
        writer.add_scalar('train_epoch/total_loss', epoch_loss, epoch_num)
        writer.add_scalar('train_epoch/loss_ce', epoch_loss_ce, epoch_num)
        writer.add_scalar('train_epoch/loss_dice', epoch_loss_dice, epoch_num)

        logging.info(
            'epoch %d, train_loss: %f, train_loss_ce: %f, train_loss_dice: %f, lr: %f' % (
                epoch_num, epoch_loss, epoch_loss_ce, epoch_loss_dice, epoch_lr
            )
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_ce = 0.0
        val_loss_dice = 0.0
        with torch.no_grad():
            for val_batch in test_loader:
                image_batch, low_res_label_batch = val_batch['img'].to(DEVICE), val_batch['masks'].to(DEVICE)  # [b, c, h, w], [b, h, w]
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
                val_loss += loss.item() * batch_size
                val_loss_ce += loss_ce.item() * batch_size
                val_loss_dice += loss_dice.item() * batch_size

        val_epoch_loss = val_loss / val_set_size
        val_epoch_loss_ce = val_loss_ce / val_set_size
        val_epoch_loss_dice = val_loss_dice / val_set_size

        writer.add_scalar('val_epoch/total_loss', val_epoch_loss, epoch_num)
        writer.add_scalar('val_epoch/loss_ce', val_epoch_loss_ce, epoch_num)
        writer.add_scalar('val_epoch/loss_dice', val_epoch_loss_dice, epoch_num)

        if val_epoch_loss_ce < best_val_loss:
            best_val_loss = val_epoch_loss_ce
            best_val_epoch = epoch_num

            save_mode_path = save_model(model, snapshot_path, "best")
            logging.info("Saved best model checkpoint to {}".format(save_mode_path))

        logging.info(
            'epoch %d, best_val_epoch: %d, val_loss: %f, val_loss_ce: %f, val_loss_dice: %f' % (
                epoch_num, best_val_epoch, val_epoch_loss, val_epoch_loss_ce, val_epoch_loss_dice
            )
        )

        save_mode_path = save_model(model, snapshot_path, "last")
        logging.info("Saved latest model checkpoint to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
