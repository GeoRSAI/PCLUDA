import sys
import os.path as osp
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append('../../..')
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from dalib.translation.randaugment import RandAugment

from dataset import get_rs_dataset, rs_dataset_name, get_rs_class_name, get_rs_dataset_imgpath


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
        return backbone
    else:
        # load models from pytorch-image-models
        print('load models from pytorch-image-models')
        return


def get_dataset(root, source, target, train_source_transform, val_transform,
                train_target_transform=None, strong_train_transform=None):

    if train_target_transform is None:
        train_target_transform = train_source_transform


    if source in rs_dataset_name and target in rs_dataset_name:
        train_source_dataset, num_classes = get_rs_dataset(root, dataset_name=source, transform=train_source_transform,
                                                           appli='train', strong_transform=None)
        train_target_dataset, _ = get_rs_dataset(root, dataset_name=target, transform=train_target_transform,
                                                 appli='train', strong_transform=strong_train_transform)
        val_dataset, _ = get_rs_dataset(root, dataset_name=target, transform=val_transform, appli='train')
        test_dataset = val_dataset
        class_names = get_rs_class_name(source)
    else:
        raise Exception('Error! Dataset is not found!')

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def get_target_dataset(root, target, val_transform):
    target_query_dataset, num_classes = get_rs_dataset_imgpath(root, dataset_name=target, transform=val_transform, appli='test')
    target_database_dataset, _ = get_rs_dataset_imgpath(root, dataset_name=target, transform=val_transform, appli='database')
    return target_query_dataset, target_database_dataset, num_classes



def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg

def get_val_dataset(dataset_name, target, val_transform):

    if dataset_name in rs_dataset_name:
        # print('target: ', target)
        target_query_dataset, n_class = get_rs_dataset_imgpath(dataset_name=target, transform=val_transform, appli='train')
        # class_names = get_rs_class_name(target)
        # num_classes = len(class_names)
    else:
        raise Exception('The dataset is not in the table!')

    return target_query_dataset

def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                        resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225),
                        strong_transform=False):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

    transforms = T.Compose(transforms)

    if strong_transform:
        print('Using strong transform！')
        transforms.transforms.insert(0, RandAugment(3, 5))

        return transforms
    else:
        return transforms


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):

    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def pretrain(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

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
