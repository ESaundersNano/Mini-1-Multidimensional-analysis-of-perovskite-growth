import hyperspy.api as hs                 # hyperspectral data processing and some useful image viewing
import h5py                               # to handle .h5 files
from matplotlib import pyplot as plt      # Plotting
import numpy as np                        # Maths
import cv2                                # Image analysis
from os import walk                       # To get filepaths automatically
from natsort import natsorted             # To easily resort file order
from datetime import datetime             # For easily parsing timestamps
import warnings
warnings.filterwarnings("ignore")         # Attempt to remove some unnecessary pyplot warnings
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import optical_image_functions as oif

def get_timestamps_hyperspec(fpaths):
    """
    Given the fpaths as a list of strings, returns the timestamps of each frame in seconds in a list of floats
    Modified for hyperspec file structure.
    """
    ts = []
    tstamps = []
    for i,fpath in enumerate(fpaths):
        # Load file
        f = h5py.File(fpath,'r')
        # Access timestamp
        tstamp = list(f['Cube/Info/Cube'].attrs['CreationDate'])[0].astype(str)
        # Remove the date and keep the time
        tstamp = tstamp.split(sep=' ')[1]
        tstamps.append(tstamp)
        # Load tstamp string into correct datetime format
        t0 = datetime.strptime(tstamps[0], "%H:%M:%S")
        t1 = datetime.strptime(tstamps[i], "%H:%M:%S")
        # Calculate difference between current frame and initial frame in seconds
        diff = t1-t0
        t = diff.total_seconds()
        # append result
        ts.append(t)
    return np.array(ts)

def extract_h5py_image_stack(fpath):
    """
    Extracts hyperspectral data from h5y file and forms it into a hyperspy object.
    Data is returned as a hyperspy 2D signal (stack of images of different wavelengths).
    Data can be easily converted to 1D signal (wavelength spectrum at each pixel) by transposing the object.
    """
    f = h5py.File(fpath,'r')
    
    pixel_size = oif.get_pixel_size(fpath)
    
    wavelengths=np.array(f['Cube/Wavelength'])
    dwavelength=wavelengths[1]-wavelengths[0]
    images = np.array(f['Cube/Images']).astype("float32")
    
    lam_size, y_size, x_size = images.shape
    
    dictlam = {'size': lam_size, 'name':'lambda', 'units':'nm', 'scale':dwavelength, 'offset':wavelengths[0]}
    dicty = {'size': y_size, 'name':'y', 'units':'µm', 'scale':pixel_size, 'offset':0} # Note y still increases as go down the image
    dictx = {'size': x_size, 'name':'x', 'units':'µm', 'scale':pixel_size, 'offset':0}

    image_stack = hs.signals.Signal2D(images, axes =[dictlam, dicty, dictx])
    return image_stack

def extract_h5py_hyperspec_data(fpaths):
    """
    Could add an option to make time a navigation axis too.
    """
    # extract images as 2D signal
    image_stacks=[]
    for fpath in fpaths:
        image_stack = extract_h5py_image_stack(fpath)
        f = h5py.File(fpath,'r')
        image_stacks.append(image_stack)
    image_stacks=np.array(image_stacks)
    
    # extract timestamps
    ts = get_timestamps_hyperspec(fpaths)
    
    # extract wavelength data from first frame
    fpath=fpaths[0]
    f = h5py.File(fpath,'r')
    wavelengths = np.array(f['Cube/Wavelength'])
    
    return image_stacks, ts, wavelengths

def convert_signal_dimensions(signal):
    """
    Have included so can change later if want to make more complicated signal e.g., with time axis included.
    """
    signal_transposed=signal.transpose()
    return signal_transposed