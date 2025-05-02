import imageio
import random
import pandas as pd
from cellpose import models, io, core, utils
from skimage.io import imread, imshow, imsave
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops, regionprops_table
import math
import napari
from skimage import measure, morphology
from scipy.ndimage import zoom
from skimage.filters import gaussian
import tifffile
import glob
import copy

print('run method figure.py')
use_GPU = core.use_gpu()

model_dir = "cropped/single_z/models/CP_20230811_153357"
a = 2

trained_model = models.CellposeModel(gpu=True, pretrained_model=model_dir)
channels = [1,2]

data = tifffile.imread("../../../cropped_cc_DAL_0044.tif")


# 2D & 025 stitching threshold
masks,_,_ = trained_model.eval(list(data), channels=channels, diameter=None, do_3D=False)
#outlines = utils.masks_to_outlines(masks)
'''
masks035 = utils.stitch3D(np.array(results), stitch_threshold=0.35)


'''
b = 3
np.save('test_method_fi.',a)
np.save('044_results',masks)

print('execution complete!')
