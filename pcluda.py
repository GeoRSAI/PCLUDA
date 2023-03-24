"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../..')
from dalib.adaptation.fixmatch import FixMatch, ImageClassifier
from dalib.adaptation.mmd import MMDLoss
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from common.utils.data import ForeverDataIterator
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.metric import accuracy, ConfusionMatrix


from common.utils.retrieval import get_rs_feature
from common.utils.retrieval.cal_precision_util import execute_retrieval

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    weakly_train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std, strong_transform=False)
    strong_train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std, strong_transform=True)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    print("weakly train transform: ", weakly_train_transform)
    print("strong train transform: ", strong_train_transform)
    print("val transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.root, args.source, args.target, weakly_train_transform, val_transform,
                                    None, strong_train_transform)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    len_source_loader = len(train_source_loader)
    len_target_loader = len(train_target_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch != 0:
        args.iters_per_epoch = n_batch

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    # 1. Maximum Mean Discrepancy Loss
    mmd_loss_fn = MMDLoss()
    # 2. Pseudo-Label Consistency Learning Loss
    pce_loss_fn = FixMatch(threshold=0.95)
    # 3.Minimize Class Confusion Loss
    mcc_loss_fn = MinimumClassConfusionLoss()

    # resume from the best checkpoint
    if args.phase != 'train':
        print('load checkpoint!')
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # 自定义的检索过程
    if args.phase == 'retrieval':
        # build the feature extractor
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)

        # build the dataset loader
        query_dataset, database_dataset, num_classes = utils.get_target_dataset(args.root, args.target, val_transform)
        query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        database_loader = DataLoader(database_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        # write image features to .h5 file
        query_index = osp.join(logger.retrieval_directory, 'test_index.h5')
        database_index = osp.join(logger.retrieval_directory, 'train_index.h5')
        get_rs_feature(query_loader, feature_extractor, device, query_index)
        get_rs_feature(database_loader, feature_extractor, device, database_index)

        # start similarity calculate with multi-pools
        ANMRR, mAP, Pk = execute_retrieval(save_path=logger.retrieval_directory, pools=10, classes=num_classes)

        print("Retrieval ANMRR = {:0.4f}".format(ANMRR))
        print("Retrieval mAP = {:0.4f}".format(mAP))
        return


    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, mmd_loss_fn, pce_loss_fn, mcc_loss_fn, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          mmd_loss_fn: nn.Module, pce_loss_fn: nn.Module, mcc_loss_fn: nn.Module, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):

    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':5.4f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # x_s: source image, labels_s: source label
        x_s, labels_s = next(train_source_iter)
        # x_t_w: target weakly-augmented sample, x_t_w: target strongly-augmented sample
        x_t_w, x_t_s, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t_w, x_t_s = x_t_w.to(device), x_t_s.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        x_t = torch.cat((x_t_w, x_t_s), dim=0)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output. y_ is the class probability output and  f_ is the feature output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)

        y_t_w, y_t_s = y_t.chunk(2, dim=0)
        f_t_w, f_t_s = f_t.chunk(2, dim=0)

        # calculate the source classification loss
        ce_loss = F.cross_entropy(y_s, labels_s)
        # calculate the feature discrepancy metric loss
        mmd_loss = mmd_loss_fn(f_s, f_t_s)
        # calculate the target pseudo-label consistency learning loss
        pce_loss = pce_loss_fn(logits_s=y_t_s, logits_w=y_t_w)
        # calculate the target minimize class confusion loss
        mcc_loss = mcc_loss_fn(y_t_w)

        loss = ce_loss + mmd_loss + args.w1 * pce_loss + args.w2 * mcc_loss

        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t_w, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(pce_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    '''
    python pcluda.py /home/sda/cross_dataset/PCLUDA_dataset/ -s UCMD -t AID -a resnet50 --epochs 30 --seed 1 --log logs/pcluda/ucmd_aid
    '''
    parser = argparse.ArgumentParser(description='DAN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-s', '--source', help='source domain')
    parser.add_argument('-t', '--target', help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--non-linear', default=False, action='store_true',
                        help='whether not use the linear version')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    # loss weight
    parser.add_argument('--w1', '--w1', default=1.0, type=float,
                        metavar='W', help='loss weight of pseudo-label consistency learning loss')
    parser.add_argument('--w2', '--w2', default=1.0, type=float,
                        metavar='W', help='loss weight of minimize class confusion loss')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis', 'retrieval', 'predict'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)

