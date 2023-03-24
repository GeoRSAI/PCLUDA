import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def cal_unknown(logits, targets):

    # compute output
    softmax_output = F.softmax(logits, dim=1)
    max_values = softmax_output.max(1).values.reshape(-1, 1)
    max_indexes = (softmax_output.argmax(1)).reshape(-1, 1)

    softmax_cat = torch.cat([softmax_output, max_values], dim=1)

    softmax_cat_np = softmax_cat.numpy()
    # 最大预测概率所在的索引
    predict_index = max_indexes.flatten().squeeze().numpy().astype(np.int32)
    # 真实标签所在类别的索引
    targets_np = targets.numpy().astype(np.int32)

    class_names = ['agricultural', 'baseball_diamond', 'beach', 'dense_residential', 'forest',
              'medium_residential', 'parking_lot', 'unknown']

    predict_class_column = [class_names[i] for i in predict_index]
    real_class_column = [class_names[i] for i in targets_np]

    softmax_df = pd.DataFrame(softmax_cat_np)
    predictlabel_df = pd.DataFrame(predict_class_column)
    target_df = pd.DataFrame(real_class_column)
    csv_data = pd.concat([softmax_df, predictlabel_df, target_df], axis=1)


    header = ['agricultural', 'baseball_diamond', 'beach', 'dense_residential', 'forest',
                            'medium_residential', 'parking_lot', 'unknown', 'max_probability', 'predict', 'target']


    csv_data.to_csv('sample.csv', header=header)

    # print(softmax_cat_np)


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    # 3个样本 4个类别
    input = torch.tensor([[10, 2, 2, 8, 1, 2, 2, 8], [6, 1, 5, 10, 3, 4, 6, 2], [2, 3, 5, 11, 5, 7, 7, 7]]).float()
    # logits_w = torch.tensor([[1, 0, 3, 8], [2, 2, 7, 0], [3, 10, 2, 1]]).float()
    targets = torch.tensor([1, 0, 3])

    output = cal_unknown(input, targets)

    # loss = unknown_bce(input)
    # print(output)
