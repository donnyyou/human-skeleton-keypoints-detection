import PIL
import pandas as pd

class Image(object):
    """
    """

    def __init__(self, id):
        # 图片路径
        path_train_im = r"./data/train_data/keypoint_train_images_20170902/"
        path_validation_im = r"./data/validation_data/keypoint_validation_images_20170911/"
        path_test_A_im = r"./data./test_data_A./keypoint_test_a_images_20170923/"
        path_test_B_im = r""

        # json路径
        path_train_json = r"./data/train_data/keypoint_train_annotations_20170909.json"
        path_validation_json = r"./data/validation_data/keypoint_validation_annotations_20170911.json"

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
        self._box = box
        self._keypoints = keypoints

    @property
    def id(self):
        return self._id

    @property
    def image(self):
        return self._image

    @property
    def dataset(self):
        return self._dataset

    def show_box(self):
        
        return

    def show_keypoints(self):