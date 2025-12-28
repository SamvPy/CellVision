import cv2
import numpy as np
import pandas as pd
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

def process_cells(
    df,
    background_map,
    path_column='path',
    key_1='folder',
    key_2='img_width'
):
    print(f'Samples to run {len(df)}')
    samples = []
    for i, row in tqdm(df.iterrows()):
        bg = background_map[(row[key_1], row[key_2])]
        cell, diff, mask = mask_cells(
            path_cell=row[path_column],
            path_background=bg,
            visualize=False
        )
        cropped_cells, _ = crop_cells(
            img_cell=cell,
            mask=mask,
            visualize=False,
        )
        cropped_cells_diff, contours = crop_cells(
            img_cell=diff,
            mask=mask,
            visualize=False,
        )
        if len(cropped_cells_diff) != 0:
            crops = {
                'index': [row['index']]*len(cropped_cells),
                'cell': cropped_cells,
                'cell_diff': cropped_cells_diff,
                'mask': [mask]*len(cropped_cells),
                'contours': contours,
            }
        else:
            crops = {
                'index': [np.nan],
                'cell': [np.nan],
                'cell_diff': [np.nan],
                'mask': [mask],
                'contours': contours,
            }

        samples.append(crops)

    return add_metrics(pd.DataFrame(samples), df)

def add_metrics(df, annotation_table):

    df['index'] = df.apply(
        lambda x: [] if isinstance(x['cell'][0], float) else [x['index']]*len(x['cell']),
        axis=1)
    df = df[df['index'].apply(len)!=0]

    columns_map = {i: col for i, col in enumerate(df.columns)}
    collapsed = df.progress_apply(lambda row: pd.DataFrame(row.tolist()).T, axis=1)
    collapsed = pd.concat(collapsed.tolist(), ignore_index=True)
    collapsed = collapsed.rename(
        columns=columns_map
    )
    collapsed['index'] = collapsed['index'].apply(lambda x: np.unique(x)[0])
    df = collapsed.merge(annotation_table)

    n_cells = df.groupby('path').count()['cell']
    df = df.merge(n_cells.reset_index(name='n'), on = 'path')

    # metrics = pd.DataFrame(df['contours'].progress_apply(lambda c: contour_metrics(c)).tolist())

    metrics_2 = pd.DataFrame(df['contours'].progress_apply(lambda c: cv2.boundingRect(c)).tolist()).rename(
        columns={
            0: 'x',
            1: 'y',
            2: 'width',
            3: 'height'
        }
    )

    df = pd.concat([df, metrics_2], axis=1)
    return df

def extract_bbox_masks(mask_matrix):
    H, W = mask_matrix.shape
    bbox_masks = {}

    mask_ids = np.unique(mask_matrix)
    mask_ids = mask_ids[mask_ids != 0]
    for mask_id in mask_ids:
        rows, cols = np.where(mask_matrix == mask_id)

        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()

        # full-size matrix filled with 0 (outside bbox)
        out = np.zeros((H, W), dtype=int)

        # copy the bounding box region from the original
        bbox = mask_matrix[r_min:r_max+1, c_min:c_max+1].copy().astype(np.int32)

        # inside the bounding box:
        # replace all values that are NOT the current mask with -1
        bbox[bbox != mask_id] = -1

        # insert back into the full-size output
        out[r_min:r_max+1, c_min:c_max+1] = bbox

        bbox_masks[mask_id] = out

    return bbox_masks

def get_bbox(mask_matrix, mask_id):
    rows, cols = np.where(mask_matrix == mask_id)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    return r_min, r_max, c_min, c_max

def crop_with_fixed_size(image, mask_matrix, mask_id, crop_size):
    """
    crop_size: (height, width)
    """
    H, W = image.shape[:2]
    ch, cw = crop_size

    r_min, r_max, c_min, c_max = get_bbox(mask_matrix, mask_id)

    # center of the bounding box
    r_center = (r_min + r_max) // 2
    c_center = (c_min + c_max) // 2

    r_start = r_center - ch // 2
    c_start = c_center - cw // 2
    r_end = r_start + ch
    c_end = c_start + cw

    # clip to image boundaries
    r_start = max(0, r_start)
    c_start = max(0, c_start)
    r_end = min(H, r_end)
    c_end = min(W, c_end)

    img_crop = image[r_start:r_end, c_start:c_end]
    mask_crop = mask_matrix[r_start:r_end, c_start:c_end]

    return img_crop, mask_crop

def crop_with_margin(image, mask_matrix, mask_id, margin):
    H, W = image.shape[:2]

    r_min, r_max, c_min, c_max = get_bbox(mask_matrix, mask_id)

    bbox_h = r_max - r_min + 1
    bbox_w = c_max - c_min + 1

    if isinstance(margin, float):
        m_h = int(bbox_h * margin)
        m_w = int(bbox_w * margin)
    else:
        m_h = m_w = int(margin)

    r_start = max(0, r_min - m_h)
    r_end   = min(H, r_max + m_h + 1)
    c_start = max(0, c_min - m_w)
    c_end   = min(W, c_max + m_w + 1)

    img_crop = image[r_start:r_end, c_start:c_end]
    mask_crop = mask_matrix[r_start:r_end, c_start:c_end]

    return img_crop, mask_crop

def boundingbox_to_crops(row):
    crops = []
    crop_masks = []
    
    for i, bounding_box in row['bounding_boxes'].items():
        crop_img, crop_mask = crop_with_margin(
            image=cv2.imread(
                row['path'],
                cv2.IMREAD_GRAYSCALE
            ),
            mask_matrix=bounding_box,
            mask_id=i,
            margin=12
        )
        crops.append(crop_img)
        crop_masks.append(crop_mask)
    return crops, crop_masks