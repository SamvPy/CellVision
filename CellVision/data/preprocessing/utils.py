import cv2
import numpy as np

def morphological_cleanup(mask, close_size=20, hole_fill=True, bridge_gaps=True):
    """
    Clean up a binary cell mask:
      - Closes small gaps and bridges close bright regions.
      - Optionally fills internal holes.
      - Optionally bridges small separations between components.
    """
    # Step 1: Closing (connect nearby white pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Step 2: Hole filling
    if hole_fill:
        # Flood fill from the edges
        h, w = mask_closed.shape
        mask_flood = mask_closed.copy()
        mask_temp = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(mask_flood, mask_temp, (0, 0), 255)
        # Invert floodfilled image
        mask_flood_inv = cv2.bitwise_not(mask_flood)
        # Combine with original to fill holes
        mask_filled = mask_closed | mask_flood_inv
    else:
        mask_filled = mask_closed

    # Step 3: Optional small-gap bridging
    if bridge_gaps:
        # Use distance transform to connect components that are within a few pixels
        dist = cv2.distanceTransform(mask_filled, cv2.DIST_L2, 5)
        # Threshold small distances to "bridge" close gaps
        gap_mask = (dist < 3).astype(np.uint8) * 255  # '3' pixels gap tolerance
        mask_bridged = cv2.dilate(mask_filled, None, iterations=1)
        mask_final = cv2.bitwise_or(mask_bridged, mask_filled)
    else:
        mask_final = mask_filled

    return mask_final

def align_to_reference(ref, img, blur_size=5, subpixel_thresh=0.01):
    """
    Align `img` to `ref` using phase correlation with integer + subpixel precision.
    Handles small translations robustly, minimizing artifacts for background subtraction.

    Parameters
    ----------
    ref : np.ndarray
        Reference grayscale image.
    img : np.ndarray
        Image to align to the reference.
    blur_size : int
        Gaussian blur kernel size to stabilize phase correlation (must be odd).
    subpixel_thresh : float
        Minimum fractional pixel shift to apply subpixel warp.

    Returns
    -------
    aligned : np.ndarray
        Aligned version of `img` with same shape as `ref`.
    """
    h, w = ref.shape
    if img.shape != ref.shape:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize intensities to reduce brightness/contrast differences
    ref_f = cv2.normalize(ref.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    img_f = cv2.normalize(img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    # Blur to suppress noise and high-frequency artifacts
    k = blur_size if blur_size % 2 == 1 else blur_size + 1
    ref_blur = cv2.GaussianBlur(ref_f, (k, k), 0)
    img_blur = cv2.GaussianBlur(img_f, (k, k), 0)

    # Compute subpixel shift using phase correlation
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    dx, dy = cv2.phaseCorrelate(ref_blur * win, img_blur * win)[0]

    # Split into integer and fractional parts
    dx_int, dy_int = int(round(dx)), int(round(dy))
    dx_frac, dy_frac = dx - dx_int, dy - dy_int

    # Integer shift using np.roll (artifact-free)
    aligned = np.roll(img, shift=(dy_int, dx_int), axis=(0,1))

    # Subpixel shift if significant
    if abs(dx_frac) > subpixel_thresh or abs(dy_frac) > subpixel_thresh:
        M = np.float32([[1, 0, dx_frac], [0, 1, dy_frac]])
        aligned = cv2.warpAffine(
            aligned,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    return aligned

def compute_stable_mask(images, diff_thresh=15):
    """
    Compute a binary mask of stable pixels across a list of aligned images.

    Parameters
    ----------
    images : list[np.ndarray]
        List of aligned grayscale images.
    diff_thresh : int
        Threshold for detecting pixel changes.

    Returns
    -------
    stable_mask : np.ndarray
        Binary mask (255 = stable pixels).
    """
    mask_union = np.zeros_like(images[0], np.uint8)
    for i in range(len(images) - 1):
        diff = cv2.absdiff(images[i], images[i + 1])
        _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        mask_union = cv2.bitwise_or(mask_union, mask)
    stable_mask = cv2.bitwise_not(mask_union)
    return stable_mask

def merge_backgrounds(*image_paths, diff_thresh=15):
    """
    Merge an arbitrary number of grayscale images to estimate a stable background.
    Automatically aligns images to the first one using phase correlation.

    Parameters
    ----------
    *image_paths : str
        Any number of image file paths.
    diff_thresh : int
        Threshold for considering a pixel 'different' across frames.

    Returns
    -------
    background : np.ndarray
        Estimated stable background image.
    """
    if len(image_paths) < 2:
        raise ValueError("Please provide at least two images.")

    # Load all images
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    if any(im is None for im in imgs):
        raise ValueError("One or more image paths could not be read.")

    # Align to first image
    ref = imgs[0]
    aligned_imgs = [ref] + [align_to_reference(ref, img) for img in imgs[1:]]

    # Compute stable/unstable regions
    stable_mask = compute_stable_mask(aligned_imgs, diff_thresh=diff_thresh)

    # Combine images
    stacked = np.stack(aligned_imgs, axis=-1).astype(np.float32)
    mean_img = np.mean(stacked, axis=-1).astype(np.uint8)
    median_img = np.median(stacked, axis=-1).astype(np.uint8)

    background = np.where(stable_mask > 0, mean_img, median_img).astype(np.uint8)
    return background

def infer_border_sizes(img_shape, target_w, target_h):
    """
    Compute symmetric (or near-symmetric) padding OR cropping
    needed to reach target size.

    If dimension is smaller → returns positive padding values.
    If dimension is larger  → returns negative padding values
      (indicating how much to crop symmetrically).

    Parameters
    ----------
    img_shape : tuple (height, width)
    target_w : int
    target_h : int

    Returns
    -------
    dict with keys: top, bottom, left, right
        Positive = pad that amount
        Negative = crop that amount
    """

    curr_h, curr_w = img_shape

    # --- WIDTH ---
    diff_w = target_w - curr_w   # positive → need pad, negative → need crop
    if diff_w >= 0:
        # pad
        left = diff_w // 2
        right = diff_w - left
    else:
        # crop (negative values indicate how much to remove)
        crop_total = -diff_w
        left = -(crop_total // 2)
        right = -(crop_total - (-left))

    # --- HEIGHT ---
    diff_h = target_h - curr_h
    if diff_h >= 0:
        top = diff_h // 2
        bottom = diff_h - top
    else:
        crop_total = -diff_h
        top = -(crop_total // 2)
        bottom = -(crop_total - (-top))

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right
    }

