import torch
from torch import nn


class DiceLoss(nn.Module):
    """
    Dice loss function to optimize semantic segmentation algorithms.

    Dice coefficient is simply the F1 score for semantic segmentation. It is calculated using formula:

        dice = (2 * intersection) / (intersection + unit)

    We assume that output and masks are type I fuzzy sets. Intersection is calculated following way:

        intersection(uA, uB) = min(uA(x), uB(x))

    Likewise, the unions is calculated following way:

        union(uA, uB) = max(uA(x), uB(x))

    As dice is always in range [0; 1], loss is equal to 1 - dice.
    This loss is better than standard cross entropy because it takes into account even the very small elements.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target) -> torch.Tensor:
        """
        Push forward and return loss
        :param output: predicted output
        :param target: target masks
        :return: loss
        """
        output = torch.squeeze(output, 0)
        target = torch.squeeze(target, 0)

        intersection = torch.min(output, target).sum((1, 2))
        union = torch.max(output, target).sum((1, 2))

        dice = (2 * intersection + 1e-8) / (intersection + union + 1e-8)
        return 1 - dice.mean()
