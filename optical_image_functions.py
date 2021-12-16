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
from scipy.interpolate import interp1d

def get_filepaths(folder):
    # Automatically get all filepaths in example folder in a sorted list
    """Given a folder address as a string, returns a list of the filepaths in the folder as strings in a sorted list"""
    fpaths = []
    _, _, fnames = next(walk(folder)) # ignores other outputs of this
    for fname in fnames:
        fpaths.append(folder+fname)
    # Automatically place into numerical order rather than 1, 10, 11
    fpaths = natsorted(fpaths)
    return fpaths

def get_h5py_contents(fpath):
    """Given the string filepath of a single h5py file, extracts and prints all the keys and attributes for inspection of data"""
    f = h5py.File(fpath,'r')
    def allkeys(obj):
        """Recursively find all keys in an h5py.Group."""
        keys = (obj.name,)
        if isinstance(obj, h5py.Group):
            for key, value in obj.items():
                if isinstance(value, h5py.Group):
                    keys = keys + allkeys(value)
                else:
                    keys = keys + (value.name,)
        return keys
    keys = allkeys(f)

    # Check each folder for any attributes and print them all out
    for key in keys:
        print(key)
        attrs = list(f[key].attrs.keys())
        if attrs != []:
            print('\t',attrs)
            
def get_timestamps(fpaths):
    """Given the fpaths as a list of strings, returns the timestamps of each frame in seconds in a list of floats"""
    ts = []
    tstamps = []
    for i,fpath in enumerate(fpaths):
        # Load file
        f = h5py.File(fpath,'r')
        # Access timestamp
        tstamp = list(f['Cube/Timestamp'])[0].astype(str)
        # Remove the date and keep the time
        tstamp = tstamp.split(sep=' ')[1]
        tstamps.append(tstamp)
        # Load tstamp string into correct datetime format
        t0 = datetime.strptime(tstamps[0], "%H:%M:%S.%f")
        t1 = datetime.strptime(tstamps[i], "%H:%M:%S.%f")
        # Calculate difference between current frame and initial frame in seconds
        diff = t1-t0
        t = diff.total_seconds()
        # append result
        ts.append(t)
    return ts

def extract_continuous_ts(ts):
    """Returns time series cut down to first continuous chunk"""
    diff0 = ts[1]-ts[0] # Original time step
    tlast=0
    i=0
    for t in ts:
        diff1= t - tlast
        if diff1 > 10*diff0: # Truncate series if the time step is larger than the original time step by a factor of 10
            return ts[:i]
        i+=1
        tlast=t
        
def extract_h5py_images(fpaths):
    """Given the fpaths of h5py files, extracts the image from each as 2D float 32 arrays"""
    images = []
    for i,fpath in enumerate(fpaths):
        # Load file
        f = h5py.File(fpath,'r')
        # Extract the image
        image = f['Cube/Images'][0]
        images.append(image)
    # Convert format for wider compatibility   
    images = np.array(images).astype('float32') # converts to 2D array with float32 values
    return images

def get_pixel_size(fpath):
    """Given fpath as a string of a suitable h5py file, returns the pixel size in um/px"""
    f = h5py.File(fpath,'r')
    scale = f['Cube/Info/Camera'].attrs['PixelSizeNm'][0] #in nm/pixel, this is the camera pixel size, not the image pixel size!!
    magnification = int(f['Cube/Info/Optics'].attrs['Objective'][0][:-1]) # need to strip 'x' symbol at end
    pixel_size = scale/(1000*magnification) # this is the image pixel calibration in um/px
    # print(r'Our image has %.3f pixels per um' % pixel_size) # should be other way around?
    return pixel_size

def get_data_h5py(folder):
    """Suggested handling of extracting data which, given a folder, returns ts, images, and pixel size"""
    fpaths = get_filepaths(folder)
    ts = get_timestamps(fpaths)
    ts=extract_continuous_ts(ts) # cuts down series if large discontinuity in time steps is detected
    images = extract_h5py_images(fpaths)
    images = images[:len(ts)] # make sure images and ts match
    pixel_size = get_pixel_size(fpaths[0])
    return ts, images, pixel_size

def blur_image(im, k_blur = 9):
    """
    Performs a Gaussian blur on an image
    @param im a 2D array, preferably of float32 values
    @param k_blur the kernel size
    @return blurred, the blurred image 2D array of float32 values
    """
    blurred = cv2.GaussianBlur(im, (k_blur, k_blur), 0)
    return blurred

def open_image(im, k_open = 9):
    """
    Opens image, eroding and then dilating the image.
    @param im a 2D array, preferably of float32 values
    @param k_open the kernel size
    @return opening, the opened image 2D array of float32 values
    """
    kernel = np.ones((k_open,k_open),np.uint8)
    opening = cv2.morphologyEx(im,cv2.MORPH_OPEN,kernel)
    return opening

def sobel_2D(im, k_sobel = 15):
    """
    Applies a sobel filter over each of the axes of a 2D array before summing result in each direction for each pixel to produce an image which has a stronger signal for gradient changes. 
    @param im a 2D array, preferably of float32 values
    @param k_sobel the kernel size
    @return opening, the opened image 2D array of float32 values
    """
    sobelx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=k_sobel)
    sobely = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=k_sobel)
    sobel = abs(sobelx) + abs(sobely)
    # sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = sobel/np.max(sobel)
    return sobel

def threshold_image(im):
    """
    
    """
    # just takes data above threshold value and makes binary it seems
    # Filters out some of the background murk
    thresh = cv2.threshold(im, np.max(im)/5, 1, cv2.THRESH_BINARY)[1].astype(np.uint8) # data type change again is preferred by cv2 for this bit for whatever reason.
    return thresh

def get_contours(binary):
    """ 
    Given a binary image, returns contours 
    Contours must be above an area/perimeter threshold
    """
    try:
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # need to ask George what going on with this next bit. Apparently was for messing around with different versions.
    except:
        try:
            _,contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours, hierarchy = cv2.findContours(binary,cv2.RETR_FLOODFILL,cv2.CHAIN_APPROX_SIMPLE)
    big_cnt = []
    areas = []
    perims = []

    for c in contours:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c,True)
        if area > 3000 or perim > 2000:
            big_cnt.append(c)
            areas.append(area)
            perims.append(perim)
#         if np.shape(c)[0] > 150:
#             big_cnt.append(c)

    return big_cnt, areas, perims