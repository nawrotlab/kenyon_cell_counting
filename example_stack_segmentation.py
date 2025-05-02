import pandas as pd
from cellpose import models, io, core
import tifffile
import glob

from cellstitch.utils import *
from cellstitch.pipeline import *
from cellstitch.evaluation import *

#get textfile names

txtfiles = []
for file in glob.glob("cropped/*.tif"):
    txtfiles.append(file)
print(txtfiles)


df = pd.DataFrame(index=txtfiles,columns = ["stitch025","stitch035","cellstitch"])

use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)

model_dir = "cropped/single_z/models/CP_20230811_153357"
trained_model = models.CellposeModel(gpu=True, pretrained_model=model_dir)
channels = [1,2]

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



#i = txtfiles[7]
i = txtfiles[-3]
print(i)
data = tifffile.imread(i)
'''
#2D & 025 stitching threshold
masks, _, _ = trained_model.eval(data, channels=channels,diameter=None,do_3D=False,stitch_threshold=0.25)
save_segmentation(masks, i+'exp_segmentation_025stitch')


#2D & 035 stitching threshold
masks, _, _ = trained_model.eval(data, channels=channels,diameter=None,do_3D=False,stitch_threshold=0.35)
save_segmentation(masks, i+'exp_segmentation_035stitch')

#CELLSTITCH
cellstitch, _, _ = trained_model.eval(list(data), channels = channels, diameter=None,do_3D=False)
cellstitch = np.array(cellstitch)

yz_masks, _, _ = trained_model.eval(list(data[:, :, :, :].transpose(2, 1, 0, 3)), channels=channels, diameter=None)
yz_masks = np.array(yz_masks).transpose(1, 0, 2)

xz_masks, _, _ = trained_model.eval(list(data[:, :, :, :].transpose(3, 1, 2, 0)), channels=channels, diameter=None)
xz_masks = np.array(xz_masks).transpose(2, 1, 0)

full_stitch(cellstitch, yz_masks, xz_masks)
save_segmentation(cellstitch, i+'exp_segmentation_cellstitch')
'''


