"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, Sequence
import torch
import torch.nn as nn

from common.modules.classifier import Classifier as ClassifierBase


__all__ = ['MMDLoss', 'ImageClassifier']


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0]) # 获取的其实是batch size的大小  2*batch_size
        # 拼接source与target的特征向量 [2*batch_size, feature_size] 其中 feature_size为特征维度、
        total = torch.cat([source, target], dim=0)
        # 在第一维增加一个维度，
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 在第二维增加一个维度
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        # 计算两个分布均值的差值，即距离; mean(0)对各列求均值，返回 1* n 矩阵 (每一列代表一个图像的向量)
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        # 等于delta的行列式的平方,即对所有向量的分布差异求和
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                # kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
                # kernel_num: 表示的是多核的数量
                # fix_sigma: 表示是否使用固定的标准差
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
            )

            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss



def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            # nn.Linear(2048, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)