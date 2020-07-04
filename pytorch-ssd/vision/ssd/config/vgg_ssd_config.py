import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 512
image_mean = np.array([85, 90, 81])  # RGB layout(85, 90, 81)#[119,120,112]
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2
#10
specs = [
    SSDSpec(64, 8, SSDBoxSizes(40, 80), [2]),
    SSDSpec(32, 16, SSDBoxSizes(80, 120), [2, 3]),
    SSDSpec(16, 32, SSDBoxSizes(120, 160), [2, 3]),
    SSDSpec(8, 64, SSDBoxSizes(160, 200), [2, 3]),
    SSDSpec(6, 124, SSDBoxSizes(240, 280), [2]),
    SSDSpec(4, 256, SSDBoxSizes(280, 320), [2])
]
# specs = [
#     SSDSpec(64, 8, SSDBoxSizes(30, 60), [2]),
#     SSDSpec(32, 16, SSDBoxSizes(60, 90), [2, 3]),
#     SSDSpec(16, 32, SSDBoxSizes(90, 130), [2, 3]),
#     SSDSpec(8, 64, SSDBoxSizes(130, 180), [2, 3]),
#     SSDSpec(6, 124, SSDBoxSizes(180, 240), [2]),
#     SSDSpec(4, 256, SSDBoxSizes(240, 300), [2])
# ]
priors = generate_ssd_priors(specs, image_size)
