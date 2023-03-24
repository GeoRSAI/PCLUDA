import torch
import torch.nn as nn

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class MinimumClassConfusionLoss(nn.Module):

    def __init__(self, temperature=2.5):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature  # default=2.5

    # 输入是一个未经激活的linear输出
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        # 为了避免拟合的时候过分关注大概率的类别，对logits的差距进行缩放 （10， 2）dist:8 --> (10/2, 2/2) dist: 4
        predictions = torch.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        # 计算概率预测的熵值，熵越大说明预测的概率分布越均匀(分类器在几个类别中摇摆不定)，熵越小说明分类器预测的结果可信度更高应给予更高的权重
        entropy_weight = entropy(predictions).detach()
        # 对熵值先取反再取对数，使得预测可信度更高的样本拥有更大的权重 (torch.exp对熵值进行了缩放)
        entropy_weight = 1 + torch.exp(-entropy_weight)
        # 对熵值权重进行归一化并乘以batch_size进行缩放，以提高权重的影响
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1

        return entropy_weight


