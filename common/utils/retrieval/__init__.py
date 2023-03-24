import torch
import h5py
import numpy as np
import os
from tqdm import tqdm
from scipy.io import savemat

def get_rs_feature(data_loader, feature_extractor, device, index_file):

    feature_extractor.eval()

    img_paths = []
    preds = []
    labels = []
    with torch.no_grad():
        for i, (img, label, img_path) in enumerate(tqdm(data_loader)):

            img = img.to(device)
            # feature = feature_extractor.get_feature(img).detach().cpu()
            feature = feature_extractor(img).detach().cpu()

            img_paths.append(img_path)
            labels.append(label)
            preds.append(feature)

        print('Writing features information to the file')

        img_paths = np.concatenate(img_paths, axis=0)
        labels = np.concatenate(labels, axis=0)
        preds = torch.cat(preds, dim=0).numpy()

        img_paths_encode = []
        for word in img_paths:
            img_paths_encode.append(word.encode())

        h5f = h5py.File(index_file, 'w')
        h5f.create_dataset('img_paths_encode', data=img_paths_encode)
        h5f.create_dataset('labels', data=labels)
        h5f.create_dataset('preds', data=preds)
        h5f.close()
        print('done!')


def get_mat_source(data_loader, feature_extractor, device, file_name):

    feature_extractor.eval()

    img_paths = []
    preds = []
    labels = []
    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm(data_loader)): # , img_path

            img = img.to(device)
            feature = feature_extractor(img).detach().cpu()#.numpy().flatten()

            # img_paths.append(img_path)
            labels.append(label)
            preds.append(feature)

        print('Writing features information to the file')

        # img_paths = np.concatenate(img_paths, axis=0)
        labels_0 = np.concatenate(labels, axis=0)  # .reshape((1, len(labels)))
        # labels_1 = labels_0 + 1
        preds = torch.cat(preds, dim=0).numpy()

        img_paths_encode = []
        # for word in img_paths:
        #     img_paths_encode.append(word.encode())



    # label_file = os.path.join(logger.retrieval_directory, 'label.mat')

    savemat(file_name, {'fea': preds, 'labels': labels_0})

    print('done!')


def get_mat_target(data_loader, feature_extractor, device, file_name):

    feature_extractor.eval()

    img_paths = []
    preds = []
    labels = []
    with torch.no_grad():
        for i, (img, label, _) in enumerate(tqdm(data_loader)): # , img_path

            img = img.to(device)
            feature = feature_extractor(img).detach().cpu()#.numpy().flatten()

            # img_paths.append(img_path)
            labels.append(label)
            preds.append(feature)

        print('Writing features information to the file')

        # img_paths = np.concatenate(img_paths, axis=0)
        labels_0 = np.concatenate(labels, axis=0)  # .reshape((1, len(labels)))
        # labels_1 = labels_0 + 1
        preds = torch.cat(preds, dim=0).numpy()

        img_paths_encode = []
        # for word in img_paths:
        #     img_paths_encode.append(word.encode())



    # label_file = os.path.join(logger.retrieval_directory, 'label.mat')

    savemat(file_name, {'fea': preds, 'labels': labels_0})

    print('done!')