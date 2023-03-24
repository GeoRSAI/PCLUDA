import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

def cal_unknown_path(val_loader, model):
    device = 'cuda:0'
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        preds = []
        targets = []
        img_names = []
        for i, (images, target, img_path) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            for i in range(len(img_path)):
                img_name = img_path[i].split('/')[-1]
                img_name = img_name.split('.')[0]

                img_names.append(img_name)

            output = model(images)
            softmax_output = F.softmax(output, dim=1)

            preds.append(softmax_output)
            targets.append(target)


        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        max_values = preds.max(1).values.reshape(-1, 1)
        max_indexes = (preds.argmax(1)).reshape(-1, 1)

        softmax_cat = torch.cat([preds, max_values], dim=1)

        softmax_cat_np = softmax_cat.cpu().numpy()
        # 最大预测概率所在的索引
        predict_index = max_indexes.flatten().squeeze().cpu().numpy().astype(np.int32)
        # 真实标签所在类别的索引
        targets_np = targets.cpu().numpy().astype(np.int32)

        class_names = ['agricultural', 'baseball_diamond', 'beach', 'dense_residential', 'forest',
                        'medium_residential', 'parking_lot', 'river', 'sparse_residential', 'storage_tanks']

        predict_class_column = [class_names[i] for i in predict_index]
        real_class_column = [class_names[i] for i in targets_np]

        img_name_df = pd.DataFrame(img_names)
        softmax_df = pd.DataFrame(softmax_cat_np)
        predictlabel_df = pd.DataFrame(predict_class_column)
        target_df = pd.DataFrame(real_class_column)
        csv_data = pd.concat([img_name_df, softmax_df, predictlabel_df, target_df], axis=1)

    return csv_data