import pandas as pd
import numpy as np

import itertools

def calc_head(req):
    # fillna
    req = req.fillna(-1)

    # keypoints intended to be used
    nose = req[0]
    neck = req[1]
    L_sho = req[2]
    R_sho = req[5]
    L_eye = req[14]
    R_eye = req[15]
    L_ear = req[16]
    R_ear = req[17]
    L = [0,14,15,16,17]
    
    if req[0] == -1:
        if L_sho != -1 and R_sho != -1:
            L_sho = [float(L_sho[1:L_sho.index(',')]),float(L_sho[L_sho.index(',')+1:len(L_sho)-1])]
            R_sho = [float(R_sho[1:R_sho.index(',')]),float(R_sho[R_sho.index(',')+1:len(R_sho)-1])]

            neck = '['+str((L_sho[0]+R_sho[0])/2)+','+str((L_sho[1]+R_sho[1])/2)+']'
        else:
            return np.nan
    else:
        neck = req[0]

    if L_ear != -1 and R_ear != -1:
        return symmetric(L_ear, R_ear, neck)
    if L_ear != -1 and nose != -1:
        return symmetric(L_ear, nose, neck)
    if nose != -1 and R_ear != -1:
        return symmetric(nose, R_ear, neck)
    if L_eye != -1 and R_eye != -1:
        return symmetric(L_eye, R_eye, neck)
    if L_eye != -1 and R_ear != -1:
        return symmetric(L_ear, R_ear, neck)
    for per in list(itertools.permutations(L,2)):
        if symmetric(req[per[0]],req[per[1]],neck) != -1:
            return symmetric(req[per[0]],req[per[1]],neck)
    return np.nan
    

def symmetric(p1,p2,p3):
    if p1 == -1 or p2 == -1 or p3 == -1:
        return np.nan
    p1 = [float(p1[1:p1.index(',')]),float(p1[p1.index(',')+1:len(p1)-1])]
    p2 = [float(p2[1:p2.index(',')]),float(p2[p2.index(',')+1:len(p2)-1])]
    p3 = [float(p3[1:p3.index(',')]),float(p3[p3.index(',')+1:len(p3)-1])]
    try:
        A = (p1[1]-p2[1])/(p1[0]-p2[0])
        C = p1[1] + p1[0]*(p2[1]-p1[1])/(p1[0]-p2[0])
        X = p3[0] - 2*A*(A*p3[0]-p3[1]+C)/(A**2+1)
        Y = p3[1] + 2*(A*p3[0]-p3[1]+C)/(A**2+1)
    except:
        return np.nan
    return [X,Y]