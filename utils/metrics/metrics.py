import torch


def fuzzy_iou(outputs: torch.Tensor, targets: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Fuzzy version of Intersection over Unit metric.
    Metric value is the ratio of intersection of the output and the target over the union of them.
    :param outputs: outputs of the model
    :param targets: true masks
    :return: mean value of iou (over all classes), intersection and union (for dice metric)
    """
    outputs = torch.squeeze(outputs, 0)
    targets = torch.squeeze(targets, 0)

    intersection = torch.min(outputs, targets).sum((1, 2))
    union = torch.max(outputs, targets).sum((1, 2))

    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.mean(), intersection, union


def dice_metric(intersection: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
    """
    Dice coefficient metric.
    Metric is simply F1 score for image segmentation.
    :param intersection: intersection of the output and the target
    :param union: union of the output and the target
    :return: mean value of dice coeff metric (over all the classes)
    """
    dice = (2 * intersection + 1e-8) / (intersection + union + 1e-8)
    return dice.mean()


def alternative_loss(dice: torch.Tensor) -> torch.Tensor:
    """
    Calculate alternative loss calculated as 1 - dice
    :param dice: dice metric
    :return: alternative loss
    """
    return 1 - dice
