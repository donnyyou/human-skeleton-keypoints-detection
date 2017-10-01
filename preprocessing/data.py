import os
import math
import functools
import random

import numpy as np
import numba
import pandas as pd

def get_human_dataframe(data="train", drop_0=True, ratio=True):
    """
    获取人物位点的DataFrame.

    Parameters
    ----------
    data: 'train' or 'validation', 需要获取的数据集.
    drop_0: 是否去除长或宽为0的样本.
    ratio: 是否将位点转为比例.

    Returns
    -------
    human: DataFrame
    """

    # 图片矩阵转为人物矩阵
    @numba.jit()
    def reshape(kp):
        human_annotations = []
        keypoint_annotations = []
        image_id = []
        for i in kp.index:
            for person in kp.loc[i, "human_annotations"].keys():
                human_annotations.append(kp.loc[i, "human_annotations"][person])
                keypoint_annotations.append(kp.loc[i, "keypoint_annotations"][person])
                image_id.append(kp.loc[i, "image_id"])
        human = pd.DataFrame({"human_annotations": human_annotations,
                              "keypoint_annotations": keypoint_annotations,
                              "image_id": image_id})
        return human

    # 这里我不太确定路径，不知道如果调用这个函数的话到底是从哪里算"./"
    # 也没敢写成绝对路径，因为上级路径可能不一样
    path_train = r"./data/train_data/keypoint_train_annotations_20170909.json"
    path_validation = r"./data/validation_data/keypoint_validation_annotations_20170911.json"

    # read data
    if data == "train":
        kp = pd.read_json(path_train)
    elif data == "validation":
        kp = pd.read_json(path_validation)
    human = reshape(kp)

    # human box
    human["box_x0"] = human["human_annotations"].apply(lambda x: x[0])
    human["box_y0"] = human["human_annotations"].apply(lambda x: x[1])
    human["box_x1"] = human["human_annotations"].apply(lambda x: x[2])
    human["box_y1"] = human["human_annotations"].apply(lambda x: x[3])

    # box size
    human["length"] = human["box_y1"] - human["box_y0"]
    human["width"] = human["box_x1"] - human["box_x0"]
    human["l/w"] = human["length"]/human["width"]

    # drop sample whose length or width is 0
    if ratio or drop_0:
        human.drop(human[(human["width"] == 0) | (human["length"] == 0)].index, inplace=True)

    # keypoint
    for i in range(1, 15):
        human["kp%d_x"%(i)] = human["keypoint_annotations"].apply(lambda x: x[3*i - 3])
        human["kp%d_y"%(i)] = human["keypoint_annotations"].apply(lambda x: x[3*i - 2])
        human["kp%d_status"%(i)] = human["keypoint_annotations"].apply(lambda x: x[3*i - 1])

    # keypoint ratio
    if ratio:
        for i in range(1, 15):
            human["kp%d_x_ratio"%(i)] = (human["kp%d_x"%(i)] - human["box_x0"]) / human["width"]
            human["kp%d_y_ratio"%(i)] = (human["kp%d_y"%(i)] - human["box_y0"]) / human["length"]

    # drop original columns
    human.drop(["human_annotations", "keypoint_annotations"], axis=1, inplace=True)

    # reset index
    human.reset_index(drop=True, inplace=True)

    return human
