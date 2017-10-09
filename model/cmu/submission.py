import os
import sys
import pandas as pd
import numpy as np
import itertools

from head import calc_head
sys.path.append("../..")
import preprocessing

# Keypoints

# AI Challenger:
# 1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，13/头顶，14/脖子

# COCO:
# [(0, 'nose'), (1, 'neck'), (2, 'Rsho'), (3, 'Relb'), (4, 'Rwri'), (5, 'Lsho'), (6, 'Lelb'), (7, 'Lwri'), 
# (8, 'Rhip'), (9, 'Rkne'), (10, 'Rank'), (11, 'Lhip'), (12, 'Lkne'), (13, 'Lank'), (14, 'Leye'), (15, 'Reye'), 
# (16, 'Lear'), (17, 'Rear'), (18, 'Head')]

def generate_result():
    # read all data and concat to a dataframe
    data = []
    for i in range(1, 601):
        data.append(pd.read_csv("./result/testA/testA_part%d.csv"%(i)))
    data = pd.concat(data, axis=0)
    data.reset_index(drop=True, inplace=True)
    data.drop("Unnamed: 0", axis=1, inplace=True)

    # calculate head point and add it to dataframe
    data.insert(loc=18, column="18", value=data.T.apply(calc_head))

    # reindex keypoints
    data = data.reindex(columns=["id", "2", "3", "4", "5", "6", "7", "8",
                                 "9", "10", "11", "12", "13", "18", "1"])
    data.columns=["image_id"]+list(range(1,15))

    # fillna with -1
    data.fillna(-1, inplace=True)

    # convert [x,y] to [x,y,v] with type(int)
    # 注意：有些[x,y]是str，有些是list，我tm也不知道为什么= =
    def convert_xy_to_xyv(xy):
        if xy == -1:
            return [0,0,0]
        elif type(xy) == list:
            return [int(i) for i in xy] + [1]
        elif type(xy) == str:
            return [int(float(i)) for i in xy[1:-1].split(",")] + [1]

    for i in range(1, 15):
        data[i] = data[i].apply(convert_xy_to_xyv)
        
    return data

def generate_submission(data):
    # sum keypoints to a list and add it to dataframe
    data["keypoints"] = data[list(range(1,15))].sum(axis=1)

    # groupby image and generate submission
    def groupby_image(image_id):
        image = data.loc[data["image_id"] == image_id, "keypoints"]
        image.index = ["human{}".format(i) for i in range(1, len(image)+1)]
        return dict(image)
    submission = pd.DataFrame(data["image_id"].unique(), columns=["image_id"])
    submission["keypoint_annotations"] = submission["image_id"].apply(groupby_image)
    
    # save
    submission.to_json("./sub.json", orient="columns")

if __name__ == "__main__":
    data = generate_result()
    generate_submission(data)