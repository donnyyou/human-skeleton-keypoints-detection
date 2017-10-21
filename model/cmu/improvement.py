"""
Improvement of original method.
"""

import pandas as pd
import numpy as np
import os
import itertools


def _match_box_and_subset(boxes, subset):
    "Matching by frequency."

    # initialize
    subset["box_id"] = None

    # function of checking point whether in the box))
    in_box = lambda p, box: (box[0] <= p[0] <= box[2]) and (box[1] <= p[1] <= box[3])

    # compute the frequencies of person's keypoints in the box
    frequency = pd.DataFrame(index=subset.index, columns=range(len(boxes)))
    for i, box in enumerate(boxes):
        frequency[i] = np.sum(subset.T.apply(lambda p: in_box(p, box)), axis=0)
    
    # match by maxing frequency
    for i in subset.index:
        subset.loc[i, "box_id"] = frequency.loc[0][frequency.loc[0] == frequency.loc[0].max()].index[0]


def _revision(boxes, subset, subset_xys, candidates):
    # some threshold
    HUMAN_THRESHOLD = 4
    PEAK_THRESHOLD = 0
    INTERSECTION_THRESHOLD = 1

    # total heatmap score
    peak_score = np.sum(subset_xys.T.apply(lambda xys: 0 if int(xys) == -1 else xys[2]), axis=0)
    subset["peak_score"] = peak_score

    for i in subset.index:

        # find all points that in the box of main person
        box_id = subset.loc[i, "box_id"]
        box = boxes[box_id]
        points = list(candidates.loc[(box[0] <= candidates[0] <= box[2]) &
                                     (box[1] <= candidates[1] <= box[3])].index)

        # filter for candidate parts
        parts = []        
        for j in subset[i+1:].index:
            # subset of points
            criterion1 = set(subset.loc[j]).issubset(set(points))
            # same main box
            criterion2 = subset.loc[j, "box_id"] == box_id
            # no intersection with the main person
            criterion3 = ((subset.loc[i]*subset.loc[j])[: 18] < 0).all()
            # threhold of score
            criterion4 = subset.loc[j, "peak_score"]/subset.loc[j, 19] > PEAK_THRESHOLD

            if criterion1 and criterion2 and criterion3 and criterion4:
                parts.append(j)
        
        # find the best combination
        max_score = 0
        appendix = None
        for i in range(1, len(parts)):
            for index in itertools.combinations(parts, i):
                person = subset.loc[parts[index]+[i], :18]
                score = subset.loc[parts[index], "peak_score"].sum()
                # check the intersection
                if not np.sum(np.sum(person > 0) <= 1) <= INTERSECTION_THRESHOLD:
                    continue
                else:
                    if score > max_score:
                        score = max_score
                        appendix = parts[index]
        
        # combination
        # TODO: max score
        # subset.loc[i, :18] = subset.loc[appendix + [i], :18].apply(max)
        subset.loc[i, 18:] = subset.loc[appendix + [i], 18:].sum()

    # generate result
    human = subset.loc[:, :18]
    for i in human:
        human = human[i].apply(lambda n: n if int(n) == -1 else list(candidates.loc[int(n), [0, 1]]))

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
        subset_coordinate[i] = subset_coordinate[i].apply(lambda n: n if int(n) == -1 else list(candidates.loc[int(n), [0, 1]]))
        subset_xys[i] = subset_xys[i].apply(lambda n: n if int(n) == -1 else list(candidates.loc[int(n), [0, 1, 2]]))

    # no box
    if not boxes:
        return subset_coordinate.T.drop([18, 19], axis=0)

    # match the box and subset
    _match_box_and_subset(boxes, subset_coordinate)
    subset["box_id"] = subset_coordinate["box_id"]
    
    # revision
    human = _revision(boxes, subset, subset_xys, candidates)

    return human