#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import shutil
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.utils.config import cfg
import time
import pickle

np.set_printoptions(suppress=True)

# write the loss information to tensorboardX 
def record_info(train_info, dev_info, iteration, logger):
    loss_info = {"train_loss": train_info['loss'], "dev_loss": dev_info['loss'], 
            "train_rpn_loss_cls": train_info['rpn_loss_cls'], "dev_rpn_loss_cls": dev_info['rpn_loss_cls'],
            "train_rpn_loss_box": train_info['rpn_loss_box'], "dev_rpn_loss_box": dev_info['rpn_loss_box'],
            "train_RCNN_loss_cls": train_info['RCNN_loss_cls'], "dev_RCNN_loss_cls": dev_info['RCNN_loss_cls'],
            "train_RCNN_loss_bbox": train_info['RCNN_loss_bbox'], "dev_RCNN_loss_bbox": dev_info['RCNN_loss_bbox'],
            "train_RCNN_loss_cls_spk": train_info['RCNN_loss_cls_spk'], "dev_RCNN_loss_cls_spk": dev_info['RCNN_loss_cls_spk']}
    logger.add_scalars("losses", loss_info, iteration)
    return 0

# Training stage.
def train(train_loader, dev_loader, model, device, optimizer, logger, log_file, args):
    # switch to train mode
    model.train()

    # define learning rate scheduler
    if args.scheduler == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.patience, verbose=True, min_lr=args.min_lr)
    elif args.scheduler == "multi":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    else:
        raise ValueError("Scheduler not defined.")

    # keep track of loss values and fg/bg numbers
    loss_rec, rpn_loss_cls_rec, rpn_loss_box_rec, RCNN_loss_cls_rec, RCNN_loss_bbox_rec = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    RCNN_loss_cls_spk_rec = AverageMeter()
    fg_cnt_rec, bg_cnt_rec = AverageMeter(), AverageMeter()
    best_score = args.best_score

    start_time = time.time() 
    for epoch in range(1, 1 + args.epochs):
        if epoch < args.start_epoch:
            continue

        iters_per_epoch = len(train_loader)

        for step, (uttname, feat, label, length) in enumerate(train_loader, 1):
            # should write in this way but not practical (data loading takes too much time)
            #if step < args.start_iter:
            #    continue

            batch_size = len(feat)

            # feat [batch_size, seq_len, feat_dim] (default: [8, 1000, 257])
            # label [batch_size, padded_len, 3] (default: [8, 20, 3], the last dimension is start_time, end_time and speaker index) 
            # length [batch_size] (label length before padding, the number of segments in the recording)
            feat, label, length = feat.to(device).float(), label.to(device).float(), length.to(device)

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_cls_spk, RCNN_loss_bbox, \
            rois_label, seg_embedding = model(feat, label, length, "train")

            # define the loss function
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() \
                    + args.alpha * RCNN_loss_cls_spk.mean() + RCNN_loss_bbox.mean()

            loss_rec.update(loss.item(), batch_size)
            rpn_loss_cls_rec.update(rpn_loss_cls.item(), batch_size)
            rpn_loss_box_rec.update(rpn_loss_box.item(), batch_size)
            RCNN_loss_cls_rec.update(RCNN_loss_cls.item(), batch_size)
            RCNN_loss_bbox_rec.update(RCNN_loss_bbox.item(), batch_size)
            RCNN_loss_cls_spk_rec.update(RCNN_loss_cls_spk.item(), batch_size)

            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
            fg_cnt_rec.update(fg_cnt.item(), batch_size)
            bg_cnt_rec.update(bg_cnt.item(), batch_size)

            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            # eval the model
            if step % args.eval_interval == 0 or step == iters_per_epoch:
                train_info = {'loss': loss_rec.avg, 'rpn_loss_cls': rpn_loss_cls_rec.avg, 'rpn_loss_box': rpn_loss_box_rec.avg, 
                        'RCNN_loss_cls': RCNN_loss_cls_rec.avg, 'RCNN_loss_bbox': RCNN_loss_bbox_rec.avg, 
                        'RCNN_loss_cls_spk': RCNN_loss_cls_spk_rec.avg, 'fg_cnt': fg_cnt_rec.avg, 'bg_cnt': bg_cnt_rec.avg}
                end_time = time.time()

                # evaluate the performance on the validation set
                start_time_valid = time.time()
                dev_info = validate(dev_loader, model, device, args)
                end_time_valid = time.time()

                model.train()
                if args.use_tfboard:
                    record_info(train_info, dev_info, (epoch - 1) * iters_per_epoch + step, logger)

                log_file.write("\nTRAIN epoch {:2d}, iter {:4d}/{:4d}, lr {:.6f}\n".format(
                    epoch, step, iters_per_epoch, optimizer.param_groups[0]['lr'])) 
                log_file.write("""TRAIN loss {:.4f}, rpn_loss_cls {:.4f}, rpn_loss_box {:.4f}, RCNN_loss_cls {:.4f}, RCNN_loss_cls_spk {:.4f}, RCNN_loss_bbox {:.4f}, fg {:.0f}, bg {:.0f}\n""".format( \
                    train_info['loss'], train_info['rpn_loss_cls'], train_info['rpn_loss_box'], 
                    train_info['RCNN_loss_cls'], train_info['RCNN_loss_cls_spk'], train_info['RCNN_loss_bbox'], train_info['fg_cnt'], train_info['bg_cnt']))
                log_file.write("TRAIN time: {:.2f}\n".format(end_time - start_time))
                log_file.flush()

                log_file.write("""VALID loss {:.4f}, rpn_loss_cls {:.4f}, rpn_loss_box {:.4f}, RCNN_loss_cls {:.4f}, RCNN_loss_cls_spk {:.4f}, RCNN_loss_bbox {:.4f}, fg {:.0f}, bg {:.0f}\n""".format( \
                    dev_info['loss'], dev_info['rpn_loss_cls'], dev_info['rpn_loss_box'], dev_info['RCNN_loss_cls'], \
                    dev_info['RCNN_loss_cls_spk'], dev_info['RCNN_loss_bbox'], dev_info['fg_cnt'], dev_info['bg_cnt']))
                log_file.write("VALID time: {:.2f}\n".format(end_time_valid - start_time_valid))
                log_file.flush()

                # adjust learning rate
                if args.scheduler == "reduce":
                    scheduler.step(dev_info['loss'])
                elif args.scheduler == "multi": 
                    scheduler.step()
                else:
                    raise ValueError("Scheduler not defined.")

                loss_rec.reset()
                rpn_loss_cls_rec.reset()
                rpn_loss_box_rec.reset()
                RCNN_loss_cls_rec.reset()
                RCNN_loss_bbox_rec.reset()
                fg_cnt_rec.reset()
                bg_cnt_rec.reset()
                RCNN_loss_cls_spk_rec.reset()
                start_time = time.time()

                # save models
                save_checkpoint({
                    'epoch': epoch,
                    'iter': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_score': best_score,
                }, "{}/model/checkpoint.pth.tar".format(args.exp_dir))
                shutil.copyfile("{}/model/checkpoint.pth.tar".format(args.exp_dir), "{}/model/epoch_{}_iter_{}.pth.tar".format(args.exp_dir, epoch, step))

                if dev_info['loss'] < best_score:
                    best_score = dev_info['loss']
                    shutil.copyfile("{}/model/checkpoint.pth.tar".format(args.exp_dir), "{}/model/modelbest.pth.tar".format(args.exp_dir))
    return 0

# Validation stage.
def validate(dev_loader, model, device, args):
    # switch to evaluate mode
    model.eval()

    loss_rec, rpn_loss_cls_rec, rpn_loss_box_rec, RCNN_loss_cls_rec, RCNN_loss_bbox_rec = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    RCNN_loss_cls_spk_rec = AverageMeter()
    fg_cnt_rec, bg_cnt_rec = AverageMeter(), AverageMeter()

    with torch.no_grad():
        for i, (uttname, feat, label, length) in enumerate(dev_loader, 1):
            batch_size = len(feat)
            feat, label, length = feat.to(device).float(), label.to(device).float(), length.to(device)
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_cls_spk, RCNN_loss_bbox, \
            rois_label, seg_embedding = model(feat, label, length, "dev")

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() \
                    + args.alpha * RCNN_loss_cls_spk.mean() + RCNN_loss_bbox.mean()

            loss_rec.update(loss.item(), batch_size)
            rpn_loss_cls_rec.update(rpn_loss_cls.item(), batch_size)
            rpn_loss_box_rec.update(rpn_loss_box.item(), batch_size)
            RCNN_loss_cls_rec.update(RCNN_loss_cls.item(), batch_size)
            RCNN_loss_bbox_rec.update(RCNN_loss_bbox.item(), batch_size)
            RCNN_loss_cls_spk_rec.update(RCNN_loss_cls_spk.item(), batch_size)

            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
            fg_cnt_rec.update(fg_cnt.item(), batch_size)
            bg_cnt_rec.update(bg_cnt.item(), batch_size)
            
    info = {'loss': loss_rec.avg, 'rpn_loss_cls': rpn_loss_cls_rec.avg, 'rpn_loss_box': rpn_loss_box_rec.avg,
            'RCNN_loss_cls': RCNN_loss_cls_rec.avg, 'RCNN_loss_bbox': RCNN_loss_bbox_rec.avg, 
            'RCNN_loss_cls_spk': RCNN_loss_cls_spk_rec.avg, 'fg_cnt': fg_cnt_rec.avg, 'bg_cnt': bg_cnt_rec.avg}
    return info

# Evaluation stage. Forward the whole recording with RPNSD model
# evaluate_no_nms will apply NMS after clustering
def evaluate_no_nms(test_loader, model, device, args):
    # switch to evaluate mode
    model.eval()
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    assert args.batch_size == 1
    with torch.no_grad():
        all_boxes = {}
        for i, (uttname, feat, _) in enumerate(test_loader, 1):
            uttname = uttname[0]
            feat = feat.to(device).float()
            batch_size, seq_len, feat_dim = feat.size(0), feat.size(1), feat.size(2)

            im_info = torch.from_numpy(np.array([[feat_dim, seq_len]]))
            im_info = im_info.expand(batch_size, im_info.size(1))

            # scale the number of region proposals according to the length of audio
            cfg.TEST.RPN_PRE_NMS_TOP_N = int(200.0 * seq_len / (10 * args.rate / args.frame_shift))
            cfg.TEST.RPN_POST_NMS_TOP_N = int(50.0 * seq_len / (10 * args.rate / args.frame_shift))
            cfg.USE_GPU_NMS = False if not args.use_gpu else True

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_cls_spk, RCNN_loss_bbox, \
            rois_label, seg_embedding = model(feat, torch.zeros(1), torch.zeros(1), "test")

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:3]
            embeddings = seg_embedding.data

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    box_deltas = box_deltas.view(-1, 2) * (torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)).to(device) \
                               + (torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)).to(device)
                    box_deltas = box_deltas.view(1, -1, 2)

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                pred_boxes = boxes.to(device)

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            embeddings = embeddings.squeeze()
            predict = torch.cat((pred_boxes, scores[:, 0:1], embeddings), 1)
            all_boxes[uttname] = predict.data.cpu().numpy()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    return 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, model_filename):
    torch.save(state, model_filename)
    return 0

if __name__ == "__main__":
    pass
