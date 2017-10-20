"""
Improvement of original method.
"""

import pandas as pd
import numpy as np
import os


def _match_box_and_subset(boxes, subset):
    "Matching by frequency."

    # initialize
    subset["box_id"] = None

    # function of checking point whether in the box
    in_box = lambda p, box: (box[0] <= p[0] <= box[2]) and (box[1] <= p[1] <= box[3])

    # compute the frequencies of person's keypoints in the box
    frequency = pd.DataFrame(index=subset.index, columns=range(len(boxes)))
    for i, box in enumerate(boxes):
        frequency[i] = np.sum(subset.T.apply(lambda p: in_box(p, box), axis=0))
    
    # match by maxing frequency
    for i in subset.index:
        # handle the situation that len(boxes) < len(subset)
        if frequency.shape[1] == 0:
            break
        # 要不要再给个阈值？
        box_id = frequency.loc[0][frequency.loc[0] == frequency.loc[0].max()].index[0]
        subset.loc[i, "box_id"] = box_id
        frequency.drop(box_id, inplace=True, axis=1)


def _revision(boxes, subset, candidates):
    # some threshold
    HUMAN_THRESHOLD = 4

    peak_score = np.sum(subset.T.apply(lambda xys: 0 if int(xys) == -1 else xys[2]), axis=0)
    num_keypoints = np.sum(subset.T.apply(lambda xys: 0 if int(xys) == -1 else 1), axis=0)
    subset["peak_score"] = peak_score
    subset["num_keypoints"] = num_keypoints

    


    return human


def detection_box_based_revision(boxes, subset, candidates):
    """
    基于detection结果的人物分割修正.

    Parameters
    ----------
    boxes: list like, [(x1, y1, x2, y2),...].
    subset: n_person * (n_keypoints + 2), contains id of candidates and score and number of visible keypoints.
    candidates: n_points * [x, y, score].

    Returns
    -------
    human: DataFrame
    """

    # convert format
    subset = pd.DataFrame(subset)
    candidates = pd.DataFrame(candidates)

    # convert id to coordinate and score
    subset_coordinate = subset.drop([18,19], axis=1)
    subset_xys = subset.drop([18,19], axis=1)
    for i in subset:
        subset_coordinate[i] = subset[i].apply(lambda n: n if int(n) == -1 else list(candidates.loc[int(n), [0, 1]]))
        subset_xys[i] = subset[i].apply(lambda n: n if int(n) == -1 else list(candidates.loc[int(n), [0, 1, 2]]))

    # no box
    if not boxes:
        return subset_coordinate.T.drop([18, 19], axis=0)

    # match the box and subset
    _match_box_and_subset(boxes, subset_coordinate)

    # revision
    human = _revision(boxes, subset_xys, candidates)

    return human