"""
The following code was developed by:

Edward Saunders, es758@cam.ac.uk
George Lewis, grl31@cam.ac.uk

This file contains the basic functions for growth rate extraction from 2D optical and hyperspectral data sets.
The code was developed for the NanoDTC mini 1 project "A multidimensional investigation into the growth conditions of halide perovskites".

08/02/2022
"""

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
    return np.array(ts)

def extract_continuous_ts(ts):
    """Returns time series cut down to first continuous chunk"""
    diff0 = ts[1]-ts[0] # Original time step
    tlast=0
    i=0
    for t in ts:
        diff1= t - tlast
        if diff1 > 10*diff0: # Truncate series if the time step is larger than the original time step by a factor of 10
            print("Discontinuous time series, data truncated")
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

def close_image(im, k_close=9):
    """
    Closes image, dilating and then eroding the image.
    @param im a 2D array, preferably of float32 values
    @param k_open the kernel size
    @return opening, the opened image 2D array of float32 values
    """
    kernel = np.ones((k_close,k_close),np.uint8)
    opening = cv2.morphologyEx(im,cv2.MORPH_CLOSE,kernel)
    return opening

def sobel_2D(im, k_sobel = 15):
    """
    Applies a sobel filter over each of the axes of a 2D array before summing result in each direction for each pixel to produce an image which has a stronger 
    signal for gradient changes. 
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

def threshold_image(im, minimum_rel_val=0.2):
    """
    Makes an image binary by making each pixel over a fraction of the maximum value present 1. 
    @param im a 2D array, preferably of float 32 values
    @param minimum_rel_val the proportion of the maximum value taken as the threshold value for polarisation
    @return thresh the 2D image array of uint8 values, all ones and zeros
    """
    # just takes data above threshold value and makes binary it seems
    # Filters out some of the background murk
    thresh = cv2.threshold(im, np.max(im)*minimum_rel_val, 1, cv2.THRESH_BINARY)[1].astype(np.uint8) # data type change again is preferred by cv2 for this bit for whatever reason.
    return thresh

def get_contours(binary):
    """ 
    Given a binary image, returns contours. Contours must be above an area/perimeter threshold.
    @param binary a 2D array image, preferably a binary image of 1s and 0s
    @return big_cnt a list of contours that are large enough
    @return areas the calculated areas of said contours
    @return perims the calculated perimeters of said contours
    """
    # Error handling statements are to attempt to account for differences in cv2 versions on some pcs.
    try:
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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

def get_contour_data(binimgs):
    """
    Basically get_contours but for a series of binary images and returns data in sub-arrays for each image.
    """
    contour_sets = []
    area_sets = []
    perim_sets = []
    for binimg in binimgs:
        contours, areas, perims = get_contours(binimg)
        contour_sets.append(contours)
        area_sets.append(areas)
        perim_sets.append(perims)
    return contour_sets, area_sets, perim_sets

def extract_area_growth_rate(area_series, time_series):
    """
    Returns growth rate by total area for each time step. Entries adjacent to NaN will be left as NaN.
    """
    gro_series = []
    index_list = np.arange(len(area_series))
    for i in index_list:
        if i == 0:
            rate = (area_series[1] - area_series[0])/(time_series[1]-time_series[0])
        elif i == index_list[-1]:
            rate = (area_series[-1] - area_series[-2])/(time_series[-1]-time_series[-2])
        else:
            rate = (area_series[i+1] - area_series[i-1])/(time_series[i+1]-time_series[i-1])
        gro_series.append(rate)
    return gro_series

def fit_speed_linear(position_array, time_array, whole_series=True, time_start=0, time_end=0, return_intercept = False):
    """
    Function used to fit gradients. Can make whole_series=False and set a time interval of interest if want to look at a subset of data but
    don't want to work out what slices to pass as array arguments.
    """
    def fit_func(x, a, b):
        return a*x + b
    def getnearpos(array,value):
        idx = np.argmin((np.abs(array-value)))
        return idx   
    if whole_series == True:
        y=position_array
        x=time_array
    else:
        i_start=getnearpos(time_array,time_start)
        i_end=getnearpos(time_array,time_end)
        y=position_array[i_start:i_end]
        x=time_array[i_start:i_end]
    params = curve_fit(fit_func, x, y)
    [a, b] = params[0]
    if return_intercept == True:
        return a, b
    else:
        return a

def fit_speed_linear_standard_dev(position_array, time_array, whole_series=True, time_start=0, time_end=0):
    """
    Function used to extract standard deviation of gradient estimates for plots. Can make whole_series=False and set a time interval of interest if want to look at a subset of data but
    don't want to work out what slices to pass as array arguments.
    """
    def fit_func(x, a, b):
        return a*x + b
    def getnearpos(array,value):
        idx = np.argmin((np.abs(array-value)))
        return idx   
    if whole_series == True:
        y=position_array
        x=time_array
    else:
        i_start=getnearpos(time_array,time_start)
        i_end=getnearpos(time_array,time_end)
        y=position_array[i_start:i_end]
        x=time_array[i_start:i_end]
    params = curve_fit(fit_func, x, y)
    return np.sqrt(params[1][0][0])

def rotate_image(im, angle):
    """
    Returns an image roated about its centre by a specified angle in degrees.
    Note that the rotation keeps the image the same size so data in the corners will be out of frame and the new corners will be black from no data.
    Warning, if rotate before taking sobel filter of image, the black corners of no data will stand out very brightly from the sobel.
    """
    row, col = im.shape
    image_centre = tuple(np.array([row, col])/2)
    rot_mat = cv2.getRotationMatrix2D(image_centre, angle, 1.0)
    result = cv2.warpAffine(im, rot_mat, (col, row), flags=cv2.INTER_LINEAR) # warpAffine takes shape tuple in reverse order to standard
    return result


def rotate_point(origin, point, angle):
    """
    Rotate a point anticlockwise by a given angle around a given origin.
    The angle should be given in degrees and the origin in index locations, which is what will be returned.
    """
    ox, oy = origin
    px, py = point
    angle=angle*np.pi/180 # convert to radians from degrees
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def onedimify(im):
    """
    One dimensionalises 2D data by averaging row values.
    Apply image preprocessing to avoid splintered peaks.
    """
    #row, col = im.shape
    im_1D_av = []
    for row in im:
        im_1D_av.append(np.average(row))
    return im_1D_av

def find_alignment(im, step_degree=1):
    """
    Rotates image until maximum 1D signal is found, which shuld correspond to the largest edge being aligned with the rows of the image.
    Step degree is in degrees. Rotates through 180 rather than 90 degrees to help the consistency of how the image will be aligned.
    """
    angles = np.arange(0, 180+step_degree, step_degree) 
    biggest_signals = []
    for angle in angles:
        rotated_image = rotate_image(im, angle)
        onedim = onedimify(rotated_image)
        biggest_signals.append(np.max(onedim))
    best_alignment = angles[biggest_signals.index(np.max(biggest_signals))]
    return best_alignment

def angle_between(v1, v2=[0,1]):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
            
            
            Changed now to give degrees
    """
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return (np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi)*180

def manual_detection(im,max_shapes=20):
    """ 
    Opens user input to manually click out pairs of points (in index positions) to define lines. Returns those pairs in lists in a list of pairs.
    """
    # Show image
    plt.imshow(im,cmap='Greys_r')
    plt.axis('off')
    all_points = []
    all_s = []
    # Get user input
    for i in range(max_shapes):
        plt.title('Line %i \n Left click to add (up to 2) points for a line \
        \n Direction assigned will be from first point to second \
        \n Right click for next line (max 20) \
        \n Middle click to remove a point (Use when zooming) \
        \n Finish by right clicking without adding points' %i,fontsize=8)
        points = plt.ginput(n=2, timeout=0, show_clicks=True,mouse_stop=3,mouse_pop=2)
        if points == []:
            break
        else:
            all_points.append(points)

            s = np.array(points)
            all_s.append(s)        
    plt.close()

    return all_s

def image_rotation_by_manual_line(im, line_points):
    """
    Given an image and a pair of points defining a line, rotates the image so the line is aligned with the vertical.
    Possible all my sign flips are solving a problem they themselves create, but I know it works right now. Originally used them to deal with problem
    of y coordinate increasing as you go down.
    """
    line=line_points[1]-line_points[0]
    line[1]=-line[1] # weird sign flip that works out
    angle=angle_between(line)
    if line_points[0][0]>line_points[1][0]: # accounts properly for sign rotation angle needs to be to align with vertical
        angle=-angle
        
    row, col = im.shape
    image_centre = tuple(np.array([row, col])/2)
    
    rotated_image = rotate_image(im, angle)
    rotated_line_points=[rotate_point(image_centre, point, -angle) for point in line_points] # Need -angle for sign convention reasons, it works
    
    return np.array(rotated_image), np.array(rotated_line_points), angle

def extract_line_data(line_points, image, crop_pixel_width =50, auto_align=False, alignment_angle=0):
    """
    Given an image and a pair of line points, will extract a 1D signal by cropping the image around those points and using onedimify.
    Image and line points should be rotated to align the line with the vertical by using image_rotation_by_manual line before use, as is done in 
    the function extract_facet_positions.
    Auto_align can be enabled which doesn't perform this primary rotation with the vertical, but adjusts the angle of the cropped image so that the
    1D data is taken perpendicular to the facet face being tracked.
    """
    # round line points to nearest integer for indexing
    line_points=np.rint(line_points)
    line_points=line_points.astype(int)
    # define crop coordinates
    x_pixel=line_points[0][0]
    y_pixel_0=line_points[0][1]
    y_pixel_1=line_points[1][1]
    # crop
    cropped=image[y_pixel_1:y_pixel_0, x_pixel-(crop_pixel_width):x_pixel+(crop_pixel_width)] #way rotation works means should always be in the right order due to direction of arrow
    
    # alignment to force perpendicular tracking
    
    if auto_align==True:
        alignment_angle=find_alignment(cropped, 0.1)
    
    cropped=rotate_image(cropped, alignment_angle)
    
    return onedimify(cropped), alignment_angle

def extract_facet_positions(line_points, outline_images, pixel_size, automatic_alignment=True, crop_window_width =50):
    """
    Returns facet positions and error arrays.
    Finds the position of a facet by finding the maximum signal in a 1D signal formed from a cropped portion of the images.
    Must pass images with edges highlighted as strength of signal at edges is used to track their position. 
    Position sign convention is based on the direction set by the line points.
    
    automatic_alignment controls alignment of the crop window rather than the whole image, but helps to extract position data taken perpendicular
    to the facet. Disable if having issues.
    
    May need to adjust crop_window_width depending on proximity of facet.
    
    Should be able to see most errors once plot position with time.
        
    Possible future changes:
    - Change facet location to be more sophisticated, perhaps using peak finder to smooth out errors and handle multiple facets in one line.
    - Check for sudden jumps in position caused by facet growing out of view of crop or new facet growing in.
    - Better integration with point selection, add a check to see if line points will be rotated out of view. 
    - Investigate why sometimes facets are just predicted to have not moved at all when clearly they do depending on cross section chosne.
    """
    first_image=outline_images[0]
    
    rotated_image, rotated_line_points, angle = image_rotation_by_manual_line(first_image, line_points)
    one_dim_facet, angle_adjustment = extract_line_data(line_points=rotated_line_points, image=rotated_image, crop_pixel_width=crop_window_width, auto_align=True)
    
    positions =[]
    error = []
    for outline in outline_images:
        rotated_image = rotate_image(outline, angle)
        onedim_facet, _ = extract_line_data(line_points=rotated_line_points, image=rotated_image, crop_pixel_width=crop_window_width, auto_align=False, alignment_angle=angle_adjustment)
        # Find position of facet in onedim
        max_signal = np.max(onedim_facet)
        position = onedim_facet.index(max_signal) # just using maximum for now
        positions.append(position)
        
        # Find error by half width at half maximum
        HM = max_signal/2
        F1 = 0
        F2 = 0
        for i in np.arange(len(onedim_facet[position:])):
            if onedim_facet[position:][i]<HM:
                F2 = i
                break
        flipped = np.flip(onedim_facet[:position+1])
        for i in np.arange(len(flipped)):
            if flipped[i]<HM:
                F1 = i
                break
        FWHM = F1 + F2
        HWHM = FWHM/2
        error.append(HWHM)
        
        
    positions = np.array(positions)
    positions -= positions[0]
    positions = pixel_size*positions
    positions = abs(positions)
    error = np.array(error)
    error = pixel_size*error
    return positions, error

"""Functions that are in notebooks which are useful but need to be tailored to each data set."""

# def get_binary(im):
#     """
#     Suggested image pre-processing to produce a binary that can get good estimates of crystal area from an image. 
#     @param im: a 2D float32 array
#     @return binary: a 2D array of uint8 values of 1s and 0s. 
#     Pre-processing can heavily influence the quality of the area estimates and should be tailored to specific sets of images.
#     Use plotting functions I have made to investigate 
#     Things to consider include:
#         Changing the order or presence of image pre-processing operations
#         Change the size of kernels from the default values suggested in optical_image_functions
#         Avoid causing area values to collapse by making the edge around the crystal discontinuous
#         Avoid having a binary that effectively has a shell in the outline
#     Use https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html as a guide for what some of the operations do.
#     Note that some of the image examples may not apply as expected depending on at which stage the operations are applied (e.g. opening base image rather than the threshold or sobel).
#     """
#     im1 = oif.blur_image(im)
#     im1=oif.close_image(im1, 8) # changing the kernel size to 8 was found to improve the end binary in this case.
#     im1=oif.open_image(im1)
#     im1=oif.sobel_2D(im1)
#     binary=oif.threshold_image(im1)
#     return binary

# def extract_best_contour_data(contour_sets, area_sets, perim_sets):
#     """
#     Suggested function for extracting the most relevant contour data. Currently just takes the set of data belonging to the contour with the biggest area. The logic being that the largest contour is
#     most likely to be around the most interesting crystal growing entirely within the view of the camera. The area was used instead of the permiter as it is more stable.
#     @param contour_sets: an array containing sets of contours for each image.
#     @param area_sets: an array containing sets (stored in arrays) of areas (floats) corresponding to contours identified in each image.
#     @param perim_sets: an array containing sets (stored in arrays) of perimeters (floats) corresponding to contours identified in each image.
#     @return best_contours, best_areas, best_perims
    
#     Possible future changes:
#     - Make more general so can pick out the contour of a single crystal entirely within frame despite other features of the image
#     - Throw an error if this isn't possible or a single crystal isn't present
#     """
#     best_contours = []
#     best_areas = []
#     best_perims = []
#     for i in np.arange(len(contour_sets)):
#         criterion = np.max(area_sets[i]) # Just takes the biggest contour
#         best_areas.append(criterion)
#         j = area_sets[i].index(criterion)
#         best_contours.append(contour_sets[i][j])
#         best_perims.append(perim_sets[i][j])
#     return best_contours, best_areas, best_perims

# def clean_area_data(contour_series, area_series, time_series, step_factor=100):
#     """
#     Suggested function for processing crystal area data before extracting growth rates. It checks if there is a large step change in the crystal being tracked. If the estimate recovers, it replaces the
#     faulty data with NaN. If the estimate does not recover by the next entry, it truncates the series in the assumption that the failure is due to the crystal reaching the edge of the field of view which
#     causes the contours to freak out a bit.
#     @param contour_series: an array of contour data for each time step
#     @param area_series: an array of area estimates (floats) for each contour
#     @param time_series: an array of timestamps (floats) for each image
#     @param step_factor: factor determining the sensitivity of the detector to a step change in area
#     @return cleaned_contour_series, cleaned_area_series, cleaned_time_series
    
#     Possible future changes:
#     - Make more lenient in truncating condition. Justified for now as possible to remove problems with image pre-processing but more flexibility may be useful for larger data sets.
#     """
#     cleaned_area_series = []
#     index_list = np.arange(len(area_series))
#     NaNcounter=0 # Counter used to track how many NaNs in total
#     for i in index_list:
#         if i == 0 or i == index_list[-1]:
#             cleaned_area_series.append(area_series[i])
#         elif i == 1:
#             cleaned_area_series.append(area_series[i])
#          # Check if something has gone wrong with area estimate
#         elif abs(area_series[i]-cleaned_area_series[i-1]) > step_factor*abs(cleaned_area_series[i-1]-cleaned_area_series[i-2]):
#             # Check if area value collapsed probably without recovery, likely due to crystal reaching edge of frame
#             if area_series[i+1] < cleaned_area_series[i-1]: # Perhaps it is a bit harsh, but checks by seeing if 2 in a row a problem
#                 break
#             # If next entry is fine, replace with a NaN and move on
#             else:
#                 cleaned_area_series.append(np.NaN)
#                 NaNcounter += 1
#         # If everything is fine, just add the area to the cleaned list
#         else:
#             cleaned_area_series.append(area_series[i])   
#     cleaned_time_series = time_series[:len(cleaned_area_series)]
#     cleaned_contour_series = contour_series[:len(cleaned_area_series)]
#     print("Area entries which are dodgy:", NaNcounter)
#     if NaNcounter > len(cleaned_area_series)*0.33:
#             print("More than a third of area data entries in cleaned range are dodgy")
#     return cleaned_contour_series, cleaned_area_series, cleaned_time_series

# def get_outline(im):
#     """
#     Suggested image pre-processing to produce an edge contrast enhanced image to be used to extract crysal facet locations from an image. 
#     @param im: a 2D float32 array
#     @return sobel: a 2D float32 array. 
#     Pre-processing can heavily influence the quality of the area estimates and should be tailored to specific sets of images.
#     Use plotting functions I have made to investigate 
#     Things to consider include:
#         Changing the order or presence of image pre-processing operations
#         Change the size of kernels from the default values suggested in optical_image_functions
#         Avoid causing area values to collapse by making the edge around the crystal discontinuous
#         Avoid having a binary that effectively has a shell in the outline
#     Use https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html as a guide for what some of the operations do.
#     Note that some of the image examples may not apply as expected depending on at which stage the operations are applied (e.g. opening base image rather than the threshold or sobel).
#     """
#     im1 = oif.blur_image(im)
#     im1=oif.close_image(im1, 8) # changing the kernel size to 8 was found to improve the end sobel filtered image for the starting data.
#     im1=oif.open_image(im1)
#     sobel=oif.sobel_2D(im1)
#     return sobel