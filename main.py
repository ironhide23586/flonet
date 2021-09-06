"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""


import cv2
import numpy as np

from data_utils.data_io import SegmapDataStreamer, StreamerContainer
import utils


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


if __name__ == '__main__':
    seg_data_streamer = SegmapDataStreamer()
    data_streamer = StreamerContainer([seg_data_streamer])
    ims, labels = data_streamer.get_data_batch()

    ims_raw = utils.nn_unpreprocess(ims)

    im = ims_raw[0]
    lbl = labels[0]
    im_edge = auto_canny(im)
    cv2.imwrite('scratchspace/x.png', im)
    cv2.imwrite('scratchspace/x_edge.png', im_edge)
    cv2.imwrite('scratchspace/y.png', lbl * 255)
    k = 0

