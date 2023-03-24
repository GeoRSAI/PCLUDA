"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
# from __future__ import print_function
# SupContrast: Supervised Contrastive Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class ThreeCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, strong_transform):
        self.transform = transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.strong_transform(x)]

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0] // 2
        features = F.normalize(features, dim=1)
        f_t1, f_t2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f_t1.unsqueeze(1), f_t2.unsqueeze(1)], dim=1)


        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)


        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 生成对角线全1，其余部分全0的二维数组: mask
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # torch.unbind:对某一个维度进行长度为1的切片，并将所有切片结果返回
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # 计算
        anchor_dot_contrast = torch.div( # torch.div: 张量和标量做逐元素除法
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # prevent computing log(0), which will produce Nan in the loss
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class simCLR(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(simCLR, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0] // 2
        features = F.normalize(features, dim=1)
        f_t1, f_t2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f_t1.unsqueeze(1), f_t2.unsqueeze(1)], dim=1)



        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # 生成对角线全1，其余部分全0的二维数组: mask
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            raise ValueError('Cannot define both `labels` and `mask`')

        contrast_count = features.shape[1]
        # torch.unbind:对某一个维度进行长度为1的切片，并将所有切片结果返回
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # torch.div: 张量和标量做逐元素除法
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # prevent computing log(0), which will produce Nan in the loss
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def test(method='SupCon'):
    criterion = SupConLoss(temperature=0.1)
    labels = torch.linspace(1,10,32).type(torch.int32)
    bsz = labels.shape[0]
    features = torch.randn(64,128)
    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)


    if method == 'SupCon':
        loss = criterion(features, labels)
    elif method == 'SimCLR':
        loss = criterion(features)

    # out = criterion(y, feat)
    # out.backward()
    # print(out.data)
    # loss.backward()
    print(loss)


if __name__ == '__main__':
    test()