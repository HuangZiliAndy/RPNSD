from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import pdb

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def generate_anchors(base_size=16, scales=2**np.arange(3, 6)):

    base_anchor = np.array([1, base_size]) - 1
    anchors = np.vstack([_scale_enum(base_anchor, scales)])
    return anchors

def _whctrs(anchor):
    w = anchor[1] - anchor[0] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    return w, x_ctr

def _mkanchors(ws, x_ctr):

    ws = ws[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         x_ctr + 0.5 * (ws - 1)))
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, x_ctr = _whctrs(anchor)
    ws = w * scales
    anchors = _mkanchors(ws, x_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    scales = np.array([1, 2, 4, 8, 16, 32, 64])
    a = generate_anchors(base_size=16, scales=scales)
    print(time.time() - t)
    print(a)
    #from IPython import embed; embed()
