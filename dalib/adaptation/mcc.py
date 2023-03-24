import torch
import torch.nn as nn
import torch.nn.functional as F

from dalib.modules.entropy import entropy

__all__ = ['MinimumClassConfusionLoss']

class MinimumClassConfusionLoss(nn.Module):
    r"""
    Minimum Class Confusion loss minimizes the class confusion in the target predictions.

    You can see more details in `Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020) <https://arxiv.org/abs/1912.03699>`_

    Args:
        temperature (float) : The temperature for rescaling, the prediction will shrink to vanilla softmax if temperature is 1.0.

    .. note::
        Make sure that temperature is larger than 0.

    Inputs: g_t
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`

    Shape:
        - g_t: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.
    """
    def __init__(self, temperature=2.5):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape

        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1

        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes

        return mcc_loss

if __name__ == '__main__':
    if __name__ == '__main__':
        loss_fn = MinimumClassConfusionLoss(temperature=2.5)

        y_pred = torch.tensor([[1.0, 2.0, 1.5],
                               [1.0, 2.0, 8.5],
                               [1.2, 1.0, 2.5],
                               ])

        # pred_entropy = entropy(y_pred)

        loss = loss_fn(y_pred)

        print('loss: ', loss)