import numpy as np
import cv2
import tqdm
from utils import *

# ------------------ Upsampling/Downsampling -----------------------------------

def downsample(image, kernel):

    """
    Downsample the image 
    
    A Gaussian filter is applied to the image, which is then downsampled by
    a factor 2
    
    Parameters
    ----------
    
    image: array-like
      original image
    kernel: array-like
      kernel of the filter
      
    Return
    ------
    
    out: array-like
      dowsampled image
    """
    
    # A CODER



def upsample(image, kernel, dst_shape=None):

    """
    Upsample the image up to the specified size
    
    The image is upsampled by first inserting zeros between adjacent pixels. 
    The resulting image is then filtered.
    
    Parameters
    ----------
    
    image: array-like
      original image
      
    kernel: array-like
      kernel of the filter
      
    dst_shape: tuple of ints
      shape of the output image
      
    Return
    ------
    
    out: array-like
      output image
    """
    
    # A CODER


# ------------------ Gaussian pyramid -----------------------------------------


def generateGaussianPyramid(image, kernel, level):

    """
    Image filtering using a Gaussian pyramid
    
    Parameters
    ----------
    
    image: array-like
      image to filter
    kernel: array-like
      convolution kernel
    level: int
      number of approximation levels in the pyramid
      
    Return
    ------
    
    output_image: array-like
      filtered image
    """

    # A CODER


# ------------------------- Temporal filter -----------------------------------

def apply_temporal_filter(images, fps, freq_range):

    """
    Apply an ideal temporal bandpass filter to the video
    
    Parameters
    ----------
    
    images: array-like
      stack of images constituting the video
      
    fps: int
      number of frames per second
      
    freq_range: tuple of float
      frequency range for the bandpass filter
      
    Return
    ------
    
    out: array-like
      filtered video represented as a stack of images
    """

    f_min, f_max = freq_range
    T = images.shape[0]
    fft_images = np.fft.fft(images, axis=0)
    freqs = np.fft.fftfreq(T, d=1/fps)

    mask = (np.abs(freqs) >= f_min) & (np.abs(freqs) <= f_max)
    mask = mask.reshape(-1, 1, 1, 1)
    
    fft_images_filtered = fft_images * mask
    
    out = np.fft.ifft(fft_images_filtered, axis=0)
    return np.real(out)


