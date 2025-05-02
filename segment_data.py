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

from cellstitch.utils import *
from cellstitch.pipeline import *
from cellstitch.evaluation import *

#get textfile names
txtfiles = []
#for file in glob.glob("cropped/*.tif"):
for file in glob.glob("../../../segment/dataset_complete/cropped/*.tif"):
    txtfiles.append(file)
txtfiles = np.sort(txtfiles)
#print(txtfiles)

#save everything
def save_segmentation(masks, name):
    label_image = np.array(masks,dtype=np.uint8)
    nlabel = len(np.unique(label_image))
    colors = [tuple(map(tuple, np.random.rand(1, 3)))[0] for i in range(0, nlabel)]
    rgb_image = color.label2rgb(label_image, colors=colors)

    img_new_t = np.transpose(rgb_image,[0, 3, 1, 2])
    img_new_t_cut = img_new_t
    img_new_t_cut_32 = img_new_t_cut.astype('float32')
    tifffile.imwrite(f"{name}.tif",
         img_new_t_cut_32,
         bigtiff=True,
         photometric='rgb',
         planarconfig='separate',
         tile=(32, 32),
         predictor=True,
         metadata={'axes': 'TZCYX'})

    np.save(name,masks)


df = pd.DataFrame(index=txtfiles,columns = ["nostitch","stitch025","stitch035","cellstitch"])

use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)

model_dir = "cropped/single_z/models/CP_20230811_153357"
trained_model = models.CellposeModel(gpu=True, pretrained_model=model_dir)
channels = [1,2]

for idx, i in enumerate(txtfiles):
    print('--------------Starting Round', idx+1, 'of', len(txtfiles),'--------------')
    data = tifffile.imread(i)
    print(i)
    #print(data.shape)

    #2D & 025 stitching threshold
    masks, _, _ = trained_model.eval(list(data), channels=channels,diameter=None,do_3D=False)
    print('len masks:',len(masks))
    tot_num = 0
    for j in range(len(masks)):
        #print('num in slice:',np.max(masks[j]))
        tot_num += np.max(masks[j])

    save_segmentation(masks, '../../../segment/testing_results/' + i[-8:-4] + 'segmented_nostitch')
    df.loc[i, 'nostitch'] = tot_num
    masks025 = utils.stitch3D(np.array(masks),stitch_threshold=0.25)
    print('-025 stitching done')
    df.loc[i, 'stitch025'] = np.max(masks025)
    save_segmentation(masks025, '../../../segment/testing_results/' + i[-8:-4] + 'segmented_stitch025')

    #2D & 035 stitching threshold
    masks035 = utils.stitch3D(np.array(masks), stitch_threshold=0.35)
    print('-035 stitching done')
    df.loc[i, 'stitch035'] = np.max(masks035)
    save_segmentation(masks035, '../../../segment/testing_results/' + i[-8:-4] + 'segmented_stitch035')


    #CELLSTITCH
    if tot_num > 10:
        cellstitch = copy.deepcopy(masks)
        cellstitch = np.array(cellstitch)
        print('-cellstitch done')

        yz_masks, _, _ = trained_model.eval(list(data[:, :, :, :].transpose(2, 1, 0, 3)), channels=channels, diameter=None)
        yz_masks = np.array(yz_masks).transpose(1, 0, 2)
        print('-yz_masks done')

        xz_masks, _, _ = trained_model.eval(list(data[:, :, :, :].transpose(3, 1, 2, 0)), channels=channels, diameter=None)
        xz_masks = np.array(xz_masks).transpose(2, 1, 0)
        print('-xz_masks done')

        full_stitch(cellstitch, yz_masks, xz_masks)
        print('-cellstitch done')
        df.loc[i, 'cellstitch'] = np.max(cellstitch)
        save_segmentation(cellstitch, '../../../segment/testing_results/' + i[-8:-4] + 'segmented_cellstitch')
    #else:
    #    df.loc[i, 'cellstitch'] = 0

print('All done!')
df.to_csv('../../../segment/testing_results/cell_counting_testing_set.csv')

print(df)
