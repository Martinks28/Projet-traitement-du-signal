from algorithms import generateGaussianPyramid, apply_temporal_filter, downsample, upsample           
from utils import *


def evm(images, fps, level, alpha, freq_range):

    """
    "Naive" Eulerian video magnification
    
    Parameters
    ----------
    
    images: nd.array
      images constituting the video sequence. The i-th frame of the video
      sequence is images[:, :, i]
      
    fps: int
      number of frames per second
      
    alpha: float
      motion amplification factor 
      
    freq_range: tuple (fmin, fmax)
      range of temporal frequencies for which the motion has to be amplified
      
    Return
    ------
    
    output_video: nd.array
      video with amplified motion
      
    """
                   
    # Filtering
    print('Filter pyramid...')                 
    filtered_images = alpha * apply_temporal_filter(images, fps, 
      freq_range=freq_range)

    # Video reconstruction
    print('Video reconstruction...')     
    output_video = np.zeros_like(images)
    for i in range(filtered_images.shape[0]):
    
        reconstructed_image = rgb2yiq(images[i]) + filtered_images[i]
        reconstructed_image = yiq2rgb(reconstructed_image)
        output_video[i] = np.clip(reconstructed_image, 0, 255)
             
    return output_video



def gaussian_evm(images, fps, level, alpha, freq_range):


    """
    Eulerian video magnification based on gaussian pyramids
    
    Parameters
    ----------
    
    images: nd.array
      images constituting the video sequence. The i-th frame of the video
      sequence is images[:, :, i]
      
    fps: int
      number of frames per second
      
    level: int
      number of decomposition levels for the pyramid
      
    alpha: float
      motion amplification factor 
      
    freq_range: tuple (fmin, fmax)
      range of temporal frequencies for which the motion has to be amplified
      
    Return
    ------
    
    output_video: nd.array
      video with amplified motion
      
    """
    
    # Gaussian pyramid
    T, H, W, C = images.shape

    shapes = []
    tmp = images[0].copy()
    for _ in range(level):
        shapes.append(tmp.shape)
        tmp = downsample(tmp, gaussian_kernel)
    approx_shape = tmp.shape

    low_res = np.zeros((T, approx_shape[0], approx_shape[1], C), dtype=np.float32)
    for t in range(T):
        low_res[t] = generateGaussianPyramid(images[t], gaussian_kernel, level)
                    
    # Filter the pyramid  
    pyramid_filter = apply_temporal_filter(low_res, fps, freq_range)         
    filtered_images = pyramid_filter * alpha
    filtered_images_high_res = np.zeros_like(images, dtype=np.float32)
    
    shapes_inverses = shapes[::-1]

    #upsample
    for t in range(T):
        tmp = filtered_images[t]
        for l in range(level):
            tmp = upsample(tmp, gaussian_kernel)
            vraie_taille = shapes_inverses[l] 
            H_cible, W_cible = vraie_taille[0], vraie_taille[1]
            tmp = tmp[:H_cible, :W_cible]
        filtered_images_high_res[t] = tmp

    # Video reconstruction   
    output_video = np.zeros_like(images)
    for i in tqdm.tqdm(range(images.shape[0]), ascii=True, desc="Video reconstruction"):
        reconstructed_image = images[i].astype(np.float32) + filtered_images_high_res[i]
        output_video[i] = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
             
    return output_video
