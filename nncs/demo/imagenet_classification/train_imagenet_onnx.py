import sys
import shutil
import os
import time
import logging
import numpy as np

abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
HOME = os.path.join(dirname, "..", "..")
sys.path.insert(0, HOME)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import nncs

from common import prepare_dataloader, adjust_learning_rate, AverageMeter, accuracy


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger is not None:
            self.logger.info("\t".join(entries))
        else:
            print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def train(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger,
    )

    # switch to train mode

    model.train()
    if args.quant:
        model.apply(nncs.fake_quantize.enable_observer)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, logger):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ",
        logger=logger
    )

    # switch to evaluate mode
    model.eval()
    if args.quant:
        model.apply(nncs.fake_quantize.disable_observer)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        # TODO: this should also be done with the ProgressMeter
        logger.info(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


def save_checkpoint(
    state, is_best, filename="checkpoint.pth.tar", bestname="model_best.pth.tar"
):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def main_worker():

    from easydict import EasyDict as edict

    args = edict()
    args.batch_size = 128
    args.workers = 4
    args.train_data = "/data/yangkang/datasets/ImageNet"
    args.val_data = "/data/yangkang/datasets/ImageNet"
    args.distributed = False
    args.gpu = 0
    args.print_freq = 20
    args.valid_float32 = True
    args.arch = "mobilenet_v2"
    args.optim = "sgd"
    args.start_epoch = 0
    args.epochs = 60
    args.lr = 1e-5
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.quant_debug = False
    args.quant = True
    args.log_file = "nncs_onnx_lr1e-5.log"
    logging.basicConfig(
        filename=args.log_file,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("logger")
    log_format = logging.Formatter("%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
    log_handler = logging.FileHandler(args.log_file)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(log_format)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)

    logger.info("start")

    model = "mobilenet-v2-71dot82.onnx"

    if args.quant:
        from nncs.fx.prepare_by_platform import (
            prepare_by_platform,
            PlatformType,
        )

        input_shape = (1, 3, 224, 224)
        model = prepare_by_platform(model, PlatformType.XNet, verify_o2t=True)

        if args.quant_debug:
            mock_t = torch.rand(input_shape).float()
            model.train()
            model.apply(nncs.fake_quantize.enable_observer)
            model(mock_t)
            model.eval()
            model.apply(nncs.fake_quantize.disable_observer)
            model.eval()
            torch.onnx.export(
                model,
                mock_t,
                "qat_graph.onnx",
                opset_version=13,
                do_constant_folding=True,
            )
            sys.exit(1)

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
            amsgrad=False,
        )

    # pylint: disable=unused-variable
    train_loader, train_sampler, val_loader, cali_loader = prepare_dataloader(args)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    model.cuda()
    if args.valid_float32:
        import copy
        val_model = copy.deepcopy(model)
        if args.quant:
            val_model.apply(nncs.fake_quantize.disable_observer)
            val_model.apply(nncs.fake_quantize.disable_fake_quant)
        validate(val_loader, val_model, criterion, args, logger)

    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, logger)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, logger)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            "{}_ckpt_lr.tar".format(args.arch),
            "{}_best_lr.tar".format(args.arch),
        )

main_worker()
