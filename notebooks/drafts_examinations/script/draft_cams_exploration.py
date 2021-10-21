#!/usr/bin/env python
# coding: utf-8

import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt


cams_dir = "/work/meteogroup/cams/"
sample_path = cams_dir+"2003/12/D20031201_00"


sample_file = Dataset(sample_path)


sample_file.variables


lats = [int(lat) for lat in sample_file["latitude"][30:71]]
# lats.reverse()
lats


lons = [int(lon) for lon in sample_file["longitude"][136:221]]
lons


sample_params = ["aod550","duaod550","aermssdul","aermssdum"]
for p in sample_params:
    print(p)
    plt.imshow(sample_file[p][0]);
    plt.show();


## from here: https://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
# import numpy as np
# import scipy.ndimage

# x = np.arange(9).reshape(3,3)

# print 'Original array:'
# print x

# print 'Resampled by a factor of 2 with nearest interpolation:'
# print scipy.ndimage.zoom(x, 2, order=0)


# print 'Resampled by a factor of 2 with bilinear interpolation:'
# print scipy.ndimage.zoom(x, 2, order=1)


# print 'Resampled by a factor of 2 with cubic interpolation:'
# print scipy.ndimage.zoom(x, 2, order=3)


original = sample_file["aod550"][0,lats,lons]
plt.imshow(original)


import scipy.ndimage
from PIL import Image

upsampled = scipy.ndimage.zoom(original, 2, order=3)
plt.imshow(upsampled)
plt.show()
print(upsampled.shape, original.shape, upsampled[0,0])
upsampled = Image.fromarray(upsampled).resize((169, 81))
upsampled = np.array(upsampled)
print(upsampled.shape, original.shape, upsampled[0,0])
plt.imshow(upsampled)
plt.show()


a = [1,2,3]
b = 4
a+[b]


original.max()





def upsample_image(original, show_result_only=False):
    upsampled = scipy.ndimage.zoom(original, 2, order=3)
    if not show_result_only:
        plt.imshow(upsampled)
        plt.show()
        print(upsampled.shape, original.shape, upsampled[0,0])
    upsampled = Image.fromarray(upsampled).resize((169, 81))
    upsampled = np.array(upsampled)
    if not show_result_only:
        print(upsampled.shape, original.shape, upsampled[0,0])
    plt.imshow(upsampled)
    plt.show()
    print(upsampled.max()-original.max())
    


sample_paths = [cams_dir+"2003/12/D20031201_00",
                cams_dir+"2003/12/D20031202_06",
                cams_dir+"2007/05/D20070501_18",
                cams_dir+"2013/06/D20130628_00",
                cams_dir+"2017/11/D20171107_06"
               ]

idx = 2

sample_file = Dataset(sample_paths[idx])

cams = [sample_file["aod550"][0,lats,lons],
        sample_file["duaod550"][0,lats,lons],
        sample_file["aermssdul"][0,lats,lons],
        sample_file["aermssdum"][0,lats,lons]
]

print([c.max() for c in cams])

for c in cams:
    upsample_image(c)








sample_paths = [cams_dir+"2005/10/D20051001_00",
                cams_dir+"2005/10/D20051001_06",
                cams_dir+"2005/10/D20051001_12",
                cams_dir+"2005/10/D20051001_18",
                cams_dir+"2005/10/D20051002_00",
               ]

for param in ["aod550", "duaod550", "aermssdul", "aermssdum"]:
    print("\n",param)
    for path in sample_paths:
        file = Dataset(path)
        original = file[param][0,lats,lons]
        upsample_image(original, show_result_only=True)




