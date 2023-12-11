from scipy.signal import convolve2d 
import warnings
import copy 
import numpy as np 
from typing import Union 
from maskfill import maskfill 
from astropy.io import fits 
import matplotlib.pyplot as plt 
from typing import Union,List,Callable

def grow_mask(mask, N):
    # Create a kernel of ones with size (2N+1, 2N+1)
    kernel_size = 2 * N + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=int)
    # Convolve the mask with the kernel
    expanded_mask = convolve2d(mask, kernel, mode='same', boundary='fill', fillvalue=0)
    # Threshold to get a boolean mask
    expanded_mask = expanded_mask > 0

    return expanded_mask

def combine_with_rejection( imagelist:List[Union[str,np.ndarray]],
                            gain: float = 1.0,
                            thresh: float = 99.5,
                            grow: int = 2,
                            combine: Callable = np.mean,
                            writesteps: bool = False):
    """Combine a series of images/spectra while rejecting cosmic rays.
    Cosmic rays identified in one frame are replaced with good pixels from the other frames. 
    

    Parameters
    ----------
    imagelist : List[Union[str,np.ndarray]]
        input images, either paths to fits files or arrays directly.
    gain : float, optional
        instrument gain for noisemodel (1.0 if not known), by default 1.0
    thresh : float, optional
        percentile on which to detect CRs in difference-frame, by default 99.5
    grow : int, optional
        grow the mask before filling, by default 2
    combine : Callable, optional
        function with which the set of images should be ultimately combined, by default np.mean
    writesteps : bool, optional
        if True, write tmp fits files with the intermediate steps, by default False

    Returns
    -------
    np.ndarray
        combined, cr-filled image. 
    """
    images = [] 
    if isinstance(imagelist[0],str):
        for i in imagelist:
            images.append(fits.getdata(i))
    else:
        for i in imagelist:
            images.append(i)
    images = np.copy(np.array(images)) 
    min_frame = np.min(images,axis=0) 
    if writesteps:
        hdu = fits.PrimaryHDU(min_frame)
        hdu.writeto('_min_frame.fits',overwrite=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        di = images - min_frame 
        noise_model = np.sqrt(min_frame*gain)/np.sqrt(gain)
        difference_images = (images - min_frame) / noise_model
        difference_images[np.isnan(difference_images)] = 0    
    if writesteps:
        hdus = [] 
        for i in difference_images: 
            hdus.append(fits.ImageHDU(i))
        hdul=fits.HDUList(hdus)
        hdul.writeto('_difference_images.fits',overwrite=True)

    masks = []
    for im,diff,d2 in zip(images,difference_images,di):
        mask_init = diff > np.nanpercentile(diff,thresh)
        mask = grow_mask(mask_init,grow)
        im[mask] -= d2[mask] 
        masks.append(mask)
    if writesteps: 
        hdus = [] 
        for i in masks:
            hdus.append(fits.ImageHDU(i))
        hdul=fits.HDUList(hdus)
        hdul.writeto('_masks.fits',overwrite=True)
    combined = combine(images,axis=0)
    return combined

def mask_and_fill(image:Union[np.ndarray,str],
                ext: int = 0,
                thresh: float = 99.9,
                grow: int = 2,
                mask: Union[np.ndarray,str] = None,
                ignore: Union[np.ndarray,str]=None,
                out_filename: str = None,
                ): 
    """
    Optionally, after combining, infill any remaining CRs based on the pixels
    surrounding them using the `maskfill` package. Because this uses a threshold
    on the final image, it *can* affect science pixels. Providing an ignore mask, 
    here `ignore`, will ensure that pixels of interest are not touched by the routine. 

    Parameters
    ----------
    image : Union[np.ndarray,str]
        combined, cr-rejected image or path to fits
    thresh : float, optional
        cutoff threshold for pixel values to be masked, by default 99.9
    grow : int, optional
        grow the generated mask, by default 2
    mask: Union[np.ndarray,str] 
        if provided, will be the exact mask passed to maskfill (overrides the threshold and grow).
    ignore : Union[np.ndarray,str], optional
        a mask identifying pixels to ignore during this routine (i.e., pixels you know are bright science pixels), by default None
    out_filename: str, optional
        if provided, a fits file will be written to the provided name, by default None.
    """
    if isinstance(image,str):
        image = fits.getdata(image,extension=ext)
    else:
        image = image 
    if mask is not None: 
        if isinstance(mask,str):
            mask = fits.getdata(mask,extension=ext)
        else:
            mask = mask.astype(int)
    else:
        mask = image > np.nanpercentile(image,thresh)
        mask = grow_mask(mask,grow).astype(int)
        if ignore is not None:
            if isinstance(ignore,str): 
                ignore = fits.getdata(ignore,extension=ext)
            else:
                ignore = ignore 
            mask[ignore.astype(bool)] = 0
            mask = mask.astype(int)
    filled = maskfill(input_image=image,mask=mask,smooth=True)        
    if out_filename is not None:
        if not out_filename.endswith('.fits'):
            out_filename+='.fits'
        hdu = fits.PrimaryHDU(filled[0])
        hdu1 = fits.ImageHDU(filled[1])
        hdu2 = fits.ImageHDU(image)
        hdu3 = fits.ImageHDU(mask)
        hdul = fits.HDUList([hdu,hdu1,hdu2,hdu3])
        hdul.writeto(out_filename,overwrite=True)
    return filled, mask 