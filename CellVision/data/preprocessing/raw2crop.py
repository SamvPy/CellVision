import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from .utils import align_to_reference, morphological_cleanup, infer_border_sizes

from glob import glob
from tqdm import tqdm
from scipy import ndimage

def mask_cells(path_cell, path_background, masking_strategy='advanced', visualize=True):
    if isinstance(path_cell, str) and os.path.exists(path_cell):
        cell = cv2.imread(path_cell, cv2.IMREAD_GRAYSCALE)
    else:
        cell = path_cell

    if isinstance(path_background, str) and os.path.exists(path_background):
        bg = cv2.imread(path_background, cv2.IMREAD_GRAYSCALE)
    else:
        bg = path_background

    # --- Step 0: Ensure equal dimensions ---
    # Ensure equal size by cropping to smallest common shape
    h = min(cell.shape[0], bg.shape[0])
    w = min(cell.shape[1], bg.shape[1])
    cell = cell[:h, :w]
    bg = bg[:h, :w]
    
    cell = align_to_reference(bg, cell)

    # --- Step 1: background subtraction ---
    diff = cv2.absdiff(cell, bg)

    # --- Step 2: thresholding ---
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # --- Step 3: advanced cleanup (your custom function) ---
    mask_cleanup = morphological_cleanup(mask)

    # --- Step 4: simple open/close cleanup ---
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations = 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_simple = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    # --- Fill black holes ---
    mask_filled = mask_simple.copy()
    h, w = mask_filled.shape[:2]
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_filled, mask_ff, (0, 0), 255)
    mask_filled_inv = cv2.bitwise_not(mask_filled)
    mask_simple = mask_simple | mask_filled_inv

    # --- Step 5: remove small objects (<15 px) ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleanup, connectivity=8)
    filtered_mask = np.zeros_like(mask_cleanup)

    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 150:
            filtered_mask[labels == i] = 255

    mask_cleanup = filtered_mask

    # --- Step 5: visualization ---
    if visualize:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        titles = [
            "Original Cell Image",
            "AbsDiff (Cell - BG)",
            "Threshold Mask",
            "Advanced Cleanup",
            "Simple Open/Close"
        ]
        images = [cell, diff, mask, mask_cleanup, mask_simple]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    if masking_strategy == 'simple':
        return cell, diff, mask_simple
    elif masking_strategy == 'advanced': 
        return cell, diff, mask_cleanup
    
    else:
        raise Exception(f"Only 'simple' and 'advanced' masking strategies are allowed.")


def crop_cells(img_cell, mask, min_area=30, margin=12, visualize=True):
    # --- find contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    h, w = img_cell.shape
    cropped_cells = []

    # --- crop each contour with margin ---
    for i, c in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(c)
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + bw + margin, w)
        y2 = min(y + bh + margin, h)
        crop_cell = img_cell[y1:y2, x1:x2]
        cropped_cells.append(crop_cell)

    # --- visualize ---
    if visualize:
        # show contours on original image
        vis = cv2.cvtColor(img_cell, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        # figure with main image + cropped cells
        if len(cropped_cells)==0:
            return
        n_show = min(len(cropped_cells), 10)
        fig, axes = plt.subplots(1, n_show + 1, figsize=(3*(n_show+1), 3))
        axes = axes.flatten()

        # show the full image with contour boxes
        axes[0].imshow(vis[..., ::-1])
        axes[0].set_title("Detected Cells")
        axes[0].axis("off")

        # show cropped cells
        for i in range(n_show):
            axes[i+1].imshow(cropped_cells[i], cmap='gray')
            axes[i+1].set_title(f"Cell {i+1}")
            axes[i+1].axis("off")

        plt.tight_layout()
        plt.show()

    return cropped_cells, contours

def pad_borders(img, target_w=68, target_h=68):
    border_additions = infer_border_sizes(
        img_shape=img.shape,
        target_w=target_w,
        target_h=target_h
    )
    img = cv2.copyMakeBorder(
        src=img,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
        **border_additions
    )