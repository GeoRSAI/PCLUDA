import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def collect_feature2(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None, use_enhance=False) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        # print('use_enhance: ', use_enhance)
        if not use_enhance:
            for i, (images, target, img_path) in enumerate(tqdm.tqdm(data_loader)):
                if max_num_features is not None and i >= max_num_features:
                    break
                # print('------------')
                # print(len(images))
                images = images.to(device)
                feature = feature_extractor(images).detach().cpu()
                all_features.append(feature)
        else:
            for i, (img_weakly, img_strong, target) in enumerate(tqdm.tqdm(data_loader)):
                if max_num_features is not None and i >= max_num_features:
                    break
                images = img_weakly.to(device)
                feature = feature_extractor(images).cpu()
                all_features.append(feature)
    return torch.cat(all_features, dim=0)

def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, return_label=False, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break

            if len(images) == 2:
                images = images[0].to(device)
            else:
                images = images.to(device)
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
            all_labels.append(target)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if return_label:
        return all_features, all_labels
    else:
        return all_features



def collect_feature_enhance(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, return_label=False, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    all_labels = []

    # iters = iter(data_loader)
    #
    # a = next(iters)

    with torch.no_grad():
        for i, ((img_weakly, img_strong), target) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            images = img_weakly.to(device)
            feature = feature_extractor(images).detach().cpu()
            all_features.append(feature)
            all_labels.append(target)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if return_label:
        return all_features, all_labels
    else:
        return all_features
