from keras.models import load_model
from glob import glob
import keras
import numpy as np
from losses import *
import random
from keras.models import Model
from extract_patches import Pipeline
from scipy.misc import imresize
from keras.utils import np_utils
import SimpleITK as sitk
import pdb
import matplotlib.pyplot as plt
import os
from scipy.ndimage.measurements import label
import cv2 
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import matplotlib.gridspec as gridspec
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap


# from evaluation_metrics import *

path_HGG = glob('/home/pi/Projects/beyondsegmentation/HGG/**')
path_LGG = glob('/home/pi/Projects/beyondsegmentation/LGG**')

test_path=glob('/home/parth/Interpretable_ML/BraTS_2018/val/**')
np.random.seed(2022)
np.random.shuffle(test_path)


def normalize_scheme(slice_not):
    '''
        normalizes each slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
    '''
    normed_slices = np.zeros(( 4,155, 240, 240))
    for slice_ix in range(4):
        normed_slices[slice_ix] = slice_not[slice_ix]
        for mode_ix in range(155):
            normed_slices[slice_ix][mode_ix] = _normalize(slice_not[slice_ix][mode_ix])

    return normed_slices    


def _normalize(slice):

    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    image_nonzero = slice[np.nonzero(slice)]

    if np.std(slice)==0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp==tmp.min()]=-9
        return tmp

def load_vol(filepath_image, model_type, slice_):

    '''
    segment the input volume
    INPUT   (1) str 'filepath_image': filepath of the volume to predict 
            (2) bool 'show': True to ,
    OUTPUt  (1) np array of the predicted volume
            (2) np array of the corresping ground truth
    '''

    #read the volume
    flair = glob( filepath_image + '/*_flair.nii.gz')
    t2 = glob( filepath_image + '/*_t2.nii.gz')
    gt = glob( filepath_image + '/*_seg.nii.gz')
    t1s = glob( filepath_image + '/*_t1.nii.gz')
    t1c = glob( filepath_image + '/*_t1ce.nii.gz')
    
    t1=[scan for scan in t1s if scan not in t1c]
    if (len(flair)+len(t2)+len(gt)+len(t1)+len(t1c))<5:
        print("there is a problem here!!! the problem lies in this patient :")
    scans_test = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i])) for i in range(len(scans_test))]


    test_im=np.array(test_im).astype(np.float32)
    test_image = test_im[0:4]
    gt=test_im[-1]
    gt[gt==4]=3

    #normalize each slice following the same scheme used for training
    test_image = normalize_scheme(test_image)

    #transform teh data to channels_last keras format
    test_image = test_image.swapaxes(0,1)
    test_image=np.transpose(test_image,(0,2,3,1))
    
    test_image, gt = np.array(test_image[slice_]), np.array(gt[slice_])
    if model_type == 'dense':
        npad = ((8, 8), (8, 8), (0, 0))
        test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=0)
        npad = ((8, 8), (8, 8))
        gt = np.pad(gt, pad_width=npad, mode='constant', constant_values=0)
    return test_image, gt

class Test_Time_Augmentation():
    
    def __init__(self):
        self.aug = iaa.SomeOf(3, [iaa.Affine(
                        rotate=iap.Normal(0.0, 3),
                        translate_px=iap.Normal(0.0, 3)),
                        iaa.AdditiveGaussianNoise(scale=0.3 * np.ptp(test_image) - 9),
                        iaa.Noop(),
                        iaa.MotionBlur(k=3, angle = [-2, 2])
                    ], random_order=True)
        
    def predict_aleatoric(self, model, test_image, iterations=1000, dropout=0.5):
        
        predictions = []
        
        for i in range(iterations):

            aug_image = aug.augment_images(test_image)
            predictions.append(model.predict(aug_image))
            
        predictions = np.array(predictions)
        
        mean = np.mean(predictions, axis = 0)
        
        var = np.var(predictions, axis = 0)
        
        print(mean.shape)
                
        plt.imshow(np.argmax(mean, axis = -1).reshape((240, 240)), vmin = 0., vmax = 3.)
        plt.show()
        plt.figure(figsize = (8, 8))
        plt.imshow(np.mean(var[:, :, :, 1:], axis = -1).reshape((240, 240)))
        plt.colorbar()
        plt.show()


if __name__ == '__main__':

    test_image, gt = load_vol(test_path[0])

    model = load_model('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/Unet_MC/UnetRes_MC.h5')
    model.load_weights('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/Unet_MC/UnetRes.60_1.066.hdf5', by_name = True)

    D = Test_Time_Augmentation()

    D.predict_aleatoric(model_res, test_image, iterations = 100, dropout = 0.)


