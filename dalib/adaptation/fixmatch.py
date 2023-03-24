import torch
import torch.nn as nn
from typing import Optional, Sequence
import torch.nn.functional as F

# from ..modules.entropy import entropy
from common.modules.classifier import Classifier as ClassifierBase


__all__ = ['FixMatch', 'ImageClassifier']


class FixMatch(nn.Module):

    def __init__(self, threshold: float):
        super(FixMatch, self).__init__()
        # The p_cutoff is the threshold value for selecting pseudo labels, the default is 0.95.
        self.p_cutoff = threshold

    def forward(self, logits_s, logits_w, name='ce'):

        if name == 'L2':
            assert logits_w.size() == logits_s.size()
            return F.mse_loss(logits_s, logits_w, reduction='mean')

        elif name == 'ce':
            num_samples = 0
            pred_w = torch.softmax(logits_w, dim=1)  # batch_size x num_classes
            max_prob, label_u = pred_w.max(1)
            mask_u = (max_prob >= self.p_cutoff).float()
            num_samples += mask_u.sum()

            pred_s = torch.softmax(logits_s, dim=1)
            loss_u = F.cross_entropy(pred_s, label_u, reduction='none')
            if num_samples == 0:
                loss = torch.tensor([0]).cuda()
            else:
                loss = (loss_u * mask_u).sum() / num_samples

            return loss


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

if __name__ == '__main__':
    # input = torch.
    input = torch.tensor([[1, 2, 2, 8], [2, 3, 5, 1], [2, 3, 5, 1]]).float()
    logits_w = torch.tensor([[1, 0, 3, 8], [2, 2, 7, 0], [3, 10, 2, 1]]).float()

    loss_fn = FixMatch(threshold=0.95)

    loss = loss_fn(logits_s=input, logits_w=logits_w)
    print(loss)
