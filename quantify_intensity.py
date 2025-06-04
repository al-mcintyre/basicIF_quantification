#!/usr/bin/env python3
# a script to calculate mean intensity per segmented region and/or expanded segmented region and save to csv 
# A McIntyre (+AI), 2025

import numpy as np
from skimage import io, measure, morphology, segmentation
from skimage.segmentation import expand_labels
from skimage.measure import regionprops
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import glob
import os

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def expand_nuclei_individually(labels, expand_pixels=5):
    expanded = expand_labels(labels, distance=expand_pixels)
    return expanded

def calculate_mean_intensities(labels, expanded_labels, image_channels, sub_bg):
    """
    Calculate the mean intensities of regions for each channel
    
    Args:
        expanded_labels (ndarray): Labeled mask of expanded nuclei.
        image_channels (list): List of multi-channel TIF images (one per channel).
    
    Returns:
        pd.DataFrame: DataFrame with cell identities, mean intensities for each channel, and centroids.
    """

    # List to hold results for each channel
    nuclear_mean_intensities = []
    expanded_mean_intensities = []
    first = True
    
    if sub_bg:
        bg = np.array(expanded_labels)
        bg[bg > 0] = 100
        bg[bg == 0] = 1
        bg[bg == 100] = 0
    
    # Calculate mean intensities in sequence for each channel
    for channel in image_channels:
        if sub_bg:
            #calculate mean background for subtraction per image per channel
            regions = measure.regionprops(bg, intensity_image = channel)
            mean_bg = [r.mean_intensity for r in regions][0]
            print('\tmean background = ',mean_bg)
        else:
            mean_bg = 0
        
        regions = measure.regionprops(labels, intensity_image = channel)
        mean_intensities = [r.mean_intensity-mean_bg for r in regions]
        nuclear_mean_intensities.append(mean_intensities)
        
        if len(expanded_labels) > 0:
            regions = measure.regionprops(expanded_labels, intensity_image = channel)
            mean_intensities = [r.mean_intensity-mean_bg for r in regions]
            expanded_mean_intensities.append(mean_intensities)
        
        if first:
            centroids = [r.centroid for r in regions]        
            first = False
    
    columns = ['Cell_ID', 'Centroid_X', 'Centroid_Y'] + \
        [f'Seg_c{i+1}_Mean_Intensity' for i in range(len(image_channels))]

    # Create the DataFrame
    if len(expanded_labels) > 0:
        data = np.column_stack([np.unique(expanded_labels)[1:], centroids] + \
                           nuclear_mean_intensities + expanded_mean_intensities)# Exclude background (0)
        columns = columns + [f'Expanded_c{i+1}_Mean_Intensity' for i in range(len(image_channels))]
    else:
        data = np.column_stack([np.unique(labels)[1:], centroids] + \
                           nuclear_mean_intensities) # Exclude background (0)
    
    df = pd.DataFrame(data, columns=columns)
    
    return df

def quantify_intensities(nuclear_label_path, channel_paths, outdir, sub_bg, condition_label='',expand_pixels=0):

    os.makedirs(outdir, exist_ok=True)

    nuclear_labels = io.imread(nuclear_label_path)
    
    # Load the multi-channel image (TIFF or PNG image with intensity values)
    image_channels = []
    for channel_path in channel_paths:
       fitype = channel_path.split('.')[-1]
       if fitype == 'tiff' or fitype == 'tif':
           image_channels.append(tifffile.imread(channel_path,is_ome=False))
       elif fitype == 'png':
           image_channels.append(plt.imread(channel_path)*65536)
       else:
           print('image type not recognized as png or tiff')
    #image_channels = [tifffile.imread(channel_path,is_ome=False) for channel_path in channel_paths]

    if expand_pixels > 0:
        expanded_labels = expand_nuclei_individually(nuclear_labels, expand_pixels)
    else:
        expanded_labels = []
    print('Calculating mean intensities for {} segmented regions'.format(np.max(nuclear_labels)))
    result_df = calculate_mean_intensities(nuclear_labels, expanded_labels, image_channels, sub_bg)
    if len(condition_label) > 0:
        result_df['condition'] = condition_label

    result_df.to_csv(outdir+'/'+nuclear_label_path.split('/')[-1].split('.')[0] + '_intensity_quantification.csv',
                    index=False)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Calculate mean intensity per nucleus and/or expanded nucleus for a list of images with associated labelled segmentations')
    parser.add_argument('-s','--segmentation_fi',type=str,required=True,help='provide file path')
    parser.add_argument('-c','--channel_fis',nargs='+',required=True,help='list of channel files with intensities to quantify (separated by spaces)')
    parser.add_argument('-o','--out_dir',type=str,required=False,help='output directory path (default = csvs/)',default='csvs')
    parser.add_argument('--exp',type=int,required=False,help='expand provided segmentations by specified number of pixels and add intensities to saved results (default = False)',default=0)
    parser.add_argument('--label',type=str,required=False,help='add column for specified condition to csv file (default = False)',default='')
    parser.add_argument('--subBG',action='store_true',required=False,help='subtract non-cellular background for quantification (default = False)',default=False)
    parser.add_argument('-v','--version',action='version',version='%(prog)s (v0.1)') 

    args = parser.parse_args()
    assert os.path.isfile(args.segmentation_fi),'no file found at {}'.format([args.segmentation_fi])
    for fi in args.channel_fis:
        assert os.path.isfile(fi),'no file found at {}'.format([fi])

    quantify_intensities(args.segmentation_fi,args.channel_fis,args.out_dir,args.subBG,args.label,args.exp)

if __name__ == "__main__":
    main()
