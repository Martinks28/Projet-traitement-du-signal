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
    
    
    img_filtred = cv2.filter2D(image, ddepth = -1, kernel = kernel)
    img_filtred = np.array(img_filtred)
    return img_filtred[::2, ::2, ::]


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
    
    H, W, C = image.shape
    image_agrandie = np.zeros((H * 2, W * 2, C), dtype=image.dtype)
    image_agrandie[::2, ::2, :] = image
    image_finale = cv2.filter2D(image_agrandie, ddepth = -1, kernel = kernel)
    image_finale = np.array(image_finale)
    return 4*image_finale


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

    current = image.copy()
    for i in range(level):
        current = downsample(current, kernel)
    return current


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


