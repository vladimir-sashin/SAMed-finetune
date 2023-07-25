import torch


def to_samed_format(labels: dict) -> dict:
    """Convert instance masks to semantic masks and normalize images"""
    instance_tensor = labels["masks"].unsqueeze(-1)
    classes_arange = torch.arange(1, labels["cls"].shape[0] + 1, device=instance_tensor.device).reshape(1, 1, 1, -1)
    instance_tensor_expanded = (instance_tensor == classes_arange)
    instance_tensor_classes = instance_tensor_expanded * (labels["cls"].flatten() + 1)
    labels["masks"] = torch.sum(instance_tensor_classes, dim=-1)

    labels["img"] = labels["img"] / 255
    return labels
