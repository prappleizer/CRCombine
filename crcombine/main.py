import argparse
import os
import warnings
from typing import List, Union

import numpy as np
from astropy.io import fits
from maskfill import maskfill
from scipy.signal import convolve2d


def grow_mask(mask, N):
    """Expand a boolean mask by N pixels in all directions."""
    kernel_size = 2 * N + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=int)
    expanded_mask = convolve2d(mask, kernel, mode="same", boundary="fill", fillvalue=0)
    expanded_mask = expanded_mask > 0
    return expanded_mask


def crcombine(
    imagelist: List[Union[str, np.ndarray]],
    gain: float = 1.0,
    thresh: float = 99.5,
    grow: int = 2,
    writesteps: bool = False,
    output_dir: str = None,
    overwrite: bool = False,
) -> List[np.ndarray]:
    """Clean cosmic rays from a set of images taken at the same pointing.

    Cosmic rays are identified in each frame by comparing to a minimum frame
    across all inputs. Masked pixels are replaced with the mean of unmasked
    pixels at that location from all other frames.

    Parameters
    ----------
    imagelist : List[Union[str, np.ndarray]]
        Input images, either paths to FITS files or arrays directly.
    gain : float, optional
        Instrument gain for noise model (1.0 if not known), by default 1.0
    thresh : float, optional
        Percentile threshold for CR detection in difference frame, by default 99.5
    grow : int, optional
        Grow the mask by this many pixels before filling, by default 2
    writesteps : bool, optional
        If True, write intermediate FITS files for debugging, by default False
    output_dir : str, optional
        If provided, cleaned frames are saved to this directory with original
        filenames. Only works when imagelist contains file paths.
    overwrite : bool, optional
        If True, overwrite existing output files. If False and output files
        exist, raises an error. By default False.

    Returns
    -------
    List[np.ndarray]
        List of cleaned images, one per input frame.
    """
    headers = []
    images = []
    filenames = []

    # Load images and headers
    if isinstance(imagelist[0], str):
        for filepath in imagelist:
            images.append(fits.getdata(filepath))
            headers.append(fits.getheader(filepath))
            filenames.append(os.path.basename(filepath))
    else:
        for img in imagelist:
            images.append(img)
            headers.append(None)
            filenames.append(None)

    images = np.array(images, dtype=float)
    n_frames = len(images)

    # Check output paths before processing
    if output_dir is not None:
        if filenames[0] is None:
            raise ValueError(
                "Cannot write output files when input is arrays. "
                "Provide FITS file paths or handle output manually."
            )
        os.makedirs(output_dir, exist_ok=True)
        output_paths = [os.path.join(output_dir, fn) for fn in filenames]
        if not overwrite:
            existing = [p for p in output_paths if os.path.exists(p)]
            if existing:
                raise FileExistsError(
                    "Output files already exist and overwrite=False:\n"
                    + "\n".join(existing)
                    + "\n\nUse --overwrite to replace, but be careful not to "
                    "overwrite your original data!"
                )

    # Create minimum frame as reference
    min_frame = np.min(images, axis=0)
    if writesteps:
        hdu = fits.PrimaryHDU(min_frame)
        hdu.writeto("_min_frame.fits", overwrite=True)

    # Compute difference images normalized by noise model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise_model = np.sqrt(np.maximum(min_frame * gain, 0)) / np.sqrt(gain)
        noise_model[noise_model == 0] = 1  # Avoid division by zero
        difference_images = (images - min_frame) / noise_model
        difference_images[np.isnan(difference_images)] = 0

    if writesteps:
        hdus = [fits.PrimaryHDU(difference_images[0])]
        for diff in difference_images[1:]:
            hdus.append(fits.ImageHDU(diff))
        fits.HDUList(hdus).writeto("_difference_images.fits", overwrite=True)

    # Generate masks for each frame
    masks = np.zeros_like(images, dtype=bool)
    for i, diff in enumerate(difference_images):
        mask_init = diff > np.nanpercentile(diff, thresh)
        masks[i] = grow_mask(mask_init, grow)

    if writesteps:
        hdus = [fits.PrimaryHDU(masks[0].astype(int))]
        for m in masks[1:]:
            hdus.append(fits.ImageHDU(m.astype(int)))
        fits.HDUList(hdus).writeto("_masks.fits", overwrite=True)

    # Create cleaned images
    cleaned = np.copy(images)

    # For each frame, replace masked pixels with mean of unmasked pixels from other frames
    for i in range(n_frames):
        mask_i = masks[i]
        if not np.any(mask_i):
            continue

        # Get indices of masked pixels in this frame
        masked_coords = np.where(mask_i)

        for y, x in zip(*masked_coords):
            # Find which other frames have clean pixels at this location
            other_frames = [j for j in range(n_frames) if j != i and not masks[j, y, x]]

            if other_frames:
                # Mean of clean pixels from other frames
                cleaned[i, y, x] = np.mean([images[j, y, x] for j in other_frames])
            else:
                # All frames have CR here; fall back to min_frame
                cleaned[i, y, x] = min_frame[y, x]

    # Write output files
    if output_dir is not None:
        for i, (cleaned_img, header, outpath) in enumerate(
            zip(cleaned, headers, output_paths)
        ):
            if header is not None:
                header = header.copy()
                header.add_history("Processed with CRCOMBINE: cosmic ray removal")
                header.add_history(f"  thresh={thresh}, grow={grow}, gain={gain}")
                header.add_history(f"  combined with {n_frames} frames")
                hdu = fits.PrimaryHDU(cleaned_img, header=header)
            else:
                hdu = fits.PrimaryHDU(cleaned_img)
            hdu.writeto(outpath, overwrite=overwrite)

    return list(cleaned)


class FilledImage:
    def __init__(self, smoothed, filled, mask):
        self.smoothed = smoothed
        self.filled = filled
        self.mask = mask


def mask_and_fill(
    image: Union[np.ndarray, str],
    ext: int = 0,
    thresh: float = 99.9,
    grow: int = 2,
    mask: Union[np.ndarray, str] = None,
    ignore: Union[np.ndarray, str] = None,
    out_filename: str = None,
):
    """
    Infill remaining CRs based on surrounding pixels using the `maskfill` package.

    This is an optional post-processing step after crcombine. Because this uses
    a threshold on the final image, it *can* affect science pixels. Providing an
    ignore mask will ensure that pixels of interest are not touched.

    Parameters
    ----------
    image : Union[np.ndarray, str]
        Cleaned image or path to FITS file
    ext : int, optional
        FITS extension to read, by default 0
    thresh : float, optional
        Percentile threshold for pixel values to be masked, by default 99.9
    grow : int, optional
        Grow the generated mask, by default 2
    mask : Union[np.ndarray, str], optional
        If provided, used as the exact mask for maskfill (overrides thresh/grow)
    ignore : Union[np.ndarray, str], optional
        Mask identifying pixels to ignore (known bright science pixels)
    out_filename : str, optional
        If provided, write output to this FITS file

    Returns
    -------
    FilledImage
        Object with .smoothed, .filled, and .mask attributes
    """
    if isinstance(image, str):
        image = fits.getdata(image, ext=ext)

    if mask is not None:
        if isinstance(mask, str):
            mask = fits.getdata(mask, ext=ext)
        mask = mask.astype(int)
    else:
        mask = image > np.nanpercentile(image, thresh)
        mask = grow_mask(mask, grow).astype(int)
        if ignore is not None:
            if isinstance(ignore, str):
                ignore = fits.getdata(ignore, ext=ext)
            mask[ignore.astype(bool)] = 0
            mask = mask.astype(int)

    filled = maskfill(input_image=image, mask=mask, smooth=True)

    if out_filename is not None:
        if not out_filename.endswith(".fits"):
            out_filename += ".fits"
        hdu = fits.PrimaryHDU(filled[0])
        hdu1 = fits.ImageHDU(filled[1])
        hdu2 = fits.ImageHDU(image)
        hdu3 = fits.ImageHDU(mask)
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3])
        hdul.writeto(out_filename, overwrite=True)

    return FilledImage(smoothed=filled[0], filled=filled[1], mask=mask)


def cli():
    parser = argparse.ArgumentParser(
        description="Remove cosmic rays from astronomical images by comparing multiple frames."
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="Input FITS images (2 or more)",
        type=str,
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for cleaned frames",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--extension",
        help="FITS extension of data (default 0)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-d",
        "--detector_gain",
        help="Detector gain for noise model (default 1.0)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-t",
        "--thresh",
        help="Percentile threshold for CR detection (default 99.5)",
        type=float,
        default=99.5,
    )
    parser.add_argument(
        "-g",
        "--grow",
        help="Grow mask by N pixels before infilling (default 2)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-w",
        "--writesteps",
        help="Write intermediate steps to FITS files for debugging",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing output files (use with caution!)",
        action="store_true",
    )

    args = parser.parse_args()

    if len(args.input) < 2:
        parser.error("At least 2 input images are required")

    # Handle FITS extension if not 0
    if args.extension != 0:
        images = [fits.getdata(f, ext=args.extension) for f in args.input]
        headers = [fits.getheader(f, ext=args.extension) for f in args.input]

        cleaned = crcombine(
            images,
            gain=args.detector_gain,
            thresh=args.thresh,
            grow=args.grow,
            writesteps=args.writesteps,
        )

        # Write outputs manually since we passed arrays
        os.makedirs(args.output, exist_ok=True)
        for img, hdr, inpath in zip(cleaned, headers, args.input):
            outpath = os.path.join(args.output, os.path.basename(inpath))
            if os.path.exists(outpath) and not args.overwrite:
                raise FileExistsError(
                    f"Output file exists: {outpath}\n"
                    "Use --overwrite to replace (but don't overwrite originals!)"
                )
            hdr = hdr.copy()
            hdr.add_history("Processed with CRCOMBINE: cosmic ray removal")
            hdr.add_history(
                f"  thresh={args.thresh}, grow={args.grow}, gain={args.detector_gain}"
            )
            hdr.add_history(f"  combined with {len(args.input)} frames")
            fits.PrimaryHDU(img, header=hdr).writeto(outpath, overwrite=args.overwrite)
    else:
        crcombine(
            args.input,
            gain=args.detector_gain,
            thresh=args.thresh,
            grow=args.grow,
            writesteps=args.writesteps,
            output_dir=args.output,
            overwrite=args.overwrite,
        )


__version__ = "0.2.0"

if __name__ == "__main__":
    cli()
