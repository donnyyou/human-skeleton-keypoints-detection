import PIL
import numpy as np
import pandas as pd
from PIL import ImageDraw

class Image(object):
    """
    """

    def __init__(self, id):
        # 图片路径
        path_train_im = r"./data/train_data/keypoint_train_images_20170902/"
        path_validation_im = r"E:/Jupyter Notebook/Data Science/AI Challenger/human-skeleton-keypoints-detection/data/validation_data/keypoint_validation_images_20170911/"
        path_test_A_im = r"./data./test_data_A./keypoint_test_a_images_20170923/"
        path_test_B_im = r""

        # json路径
        path_train_json = r"./data/train_data/keypoint_train_annotations_20170909.json"
        path_validation_json = r"E:/Jupyter Notebook/Data Science/AI Challenger/human-skeleton-keypoints-detection/data/validation_data/keypoint_validation_annotations_20170911.json"

        # 查找图片并打开
        for dataset, path in enumerate([path_train_im, path_validation_im, path_test_A_im, path_test_B_im]):
            try:
                image = PIL.Image.open(path + id +".jpg")
            except:
                continue
            break
        else:
            raise Exception("Can't find image: {id}".format(id="id"))

        # 读取对应的json，并获取位点信息
        dataset = {0: "train", 1: "validation", 2: "test_A", 3: "test_B"}[dataset]
        if dataset in ["train", "validation"]:
            df = pd.read_json(path_train_json) if dataset == "train" else pd.read_json(path_validation_json)
            box = df.loc[df["image_id"] == id, "human_annotations"]
            keypoints = df.loc[df["image_id"] == id, "keypoint_annotations"]
        else:
            box = None
            keypoints = None
        
        self._id = id
        self._image = image
        self._dataset = dataset
        self._box = box.values[0]
        self._keypoints = keypoints.values[0]

    @property
    def id(self):
        return self._id

    @property
    def image(self):
        return self._image

    @property
    def dataset(self):
        return self._dataset

    @property
    def box(self):
        return self._box

    @property
    def keypoints(self):
        return self._keypoints

    def show_box(self, line_width=4, fill=(255, 0, 0)):
        image_box = self._image.copy()
        draw = ImageDraw.Draw(image_box)
        for human in self._box.keys():
            point1 = self._box[human][: 2]
            point2 = self._box[human][2: ]
            point3 = [point1[0], point2[1]]
            point4 = [point2[0], point1[1]]
            for p1, p2 in [(point1, point3), (point1, point4), \
                           (point2, point3), (point2, point4)]:
                draw.line(p1 + p2, width=line_width, fill=fill)
        return image_box

    def show_keypoints(self, line=True, kp_size=3, line_width=3, point_fill=(255, 0, 0), line_fill=(255, 0, 0)):
        image_kp = self._image.copy()
        draw = ImageDraw.Draw(image_kp)

        for human in self._keypoints.keys():
            # draw keypoints
            kp = np.reshape(self._keypoints[human], (14, 3))
            for i in range(14):
                if kp[i, 2] == 1:
                    draw.rectangle([kp[i, 0] - kp_size, kp[i, 1] - kp_size, \
                                    kp[i, 0] + kp_size, kp[i, 1] + kp_size], \
                                    outline=point_fill, fill=point_fill)
            # draw lines
            if line:
                # line between keypoints
                line_list = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], \
                                      [9, 10], [10, 11], [12, 13], [0, 13], [3, 13], \
                                      [0, 6], [3, 9], [6, 9]])
                for i in range(line_list.shape[0]):
                    if kp[line_list[i, 0], 2] == 1 and kp[line_list[i, 1], 2] == 1:
                        draw.line([(kp[line_list[i, 0], 0], kp[line_list[i, 0], 1]), \
                                   (kp[line_list[i, 1], 0], kp[line_list[i, 1], 1])], \
                                   width=line_width, fill=line_fill)
        return image_kp
