import pandas as pd
import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import cv2
import os
import lance
import torch

io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")

tqdm.pandas()

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

def main():
  
    paths = pd.read_csv('/home/sam/cellenone_project/CellVision/notebooks/iteration2/selected_images.csv')
    _ = paths.pop('Unnamed: 0')
    paths.shape

    # Read in the images
    images = []
    for path in tqdm(paths['path'].tolist()):
        images.append(
            io.imread(path)[..., np.newaxis]
        )

    # Remove 3D images
    idxes = [
        i for i, img in enumerate(images) if img.shape[2]==1
    ]
    paths = paths.loc[idxes,:]
    images_selected = [
        img for i, img in enumerate(images) if i in idxes
    ]

    # Load the model
    model = models.CellposeModel(
        gpu=True,
        pretrained_model='/home/sam/cellenone_project/CellVision/notebooks/iteration2/models/CPSAM_CellenONE',
    )


    # Segment the cells from the images
    flow_threshold = 0.4
    cellprob_threshold = 0.0

    masks, flows, styles = model.eval(
        images_selected,
        batch_size=32,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )

    df = paths.copy()
    df['CP:mask'] = masks

    # Process the masks
    df_masks = pd.DataFrame({'masks': masks})
    df_masks['bounding_boxes'] = df_masks['masks'].apply(lambda x: extract_bbox_masks(x))
    df_masks['n'] = df_masks['bounding_boxes'].apply(len)

    df_all = pd.concat([df.reset_index(drop=True), df_masks.reset_index(drop=True)], axis=1)
    df_all = df_all.reset_index(drop=True).reset_index()

    crops = df_all.loc[:, ['index', 'path', 'bounding_boxes']].dropna(subset=['path', 'bounding_boxes'], axis=0)
    crop_and_mask = crops.progress_apply(
        boundingbox_to_crops,
        axis=1
    )
    crop_and_mask = crop_and_mask.reset_index(name='crop_and_mask')

    crop_and_mask['crop_with_margin'] = crop_and_mask['crop_and_mask'].apply(lambda x: x[0])
    crop_and_mask['crop_mask'] = crop_and_mask['crop_and_mask'].apply(lambda x: x[1])

    df_all['crop_with_margin'] = crop_and_mask['crop_with_margin']
    df_all = df_all.explode('crop_with_margin')
    df_all['crop_mask'] = crop_and_mask['crop_mask'].explode(ignore_index=True)
    df_all = df_all.reset_index(drop=True)
    df_all['crop_mask'] = crop_and_mask['crop_mask'].explode(ignore_index=True)

    df_all['crop_xlim'] = df_all['crop_with_margin'].apply(lambda x: np.array(x).shape[0] if not isinstance(x, float) else np.nan)
    df_all['crop_ylim'] = df_all['crop_with_margin'].apply(lambda x: np.array(x).shape[1] if not isinstance(x, float) else np.nan)

    df_all['x-y ratio'] = df_all.apply(lambda x: x['crop_xlim']/x['crop_ylim'], axis=1)

    df_all = df_all.reset_index(drop=True)

    # Filter the crops
    n = 6
    min_ = .75
    max_ = 1.2
    final_selection = df_all[
        (df_all['n']!=0) &
        (df_all['n']<n) &
        (df_all['x-y ratio'] < max_) &
        (df_all['x-y ratio'] > min_)
    ]

    _ = final_selection.pop('CP:mask')
    _ = final_selection.pop('bounding_boxes')

    for col in ['masks', 'crop_with_margin', 'crop_mask']:
        final_selection[col] = final_selection[col].apply(lambda x: x.astype(float).tolist())


    root_data_storage = '/public/conode53/sam/lance/cellenONE'

    print('Writing...')
    lance.write_dataset(
        data_obj=final_selection,
        uri=os.path.join(
            root_data_storage,
            'CellVision_CellPose_1.lance'
        )
    )

if __name__ == '__main__':
    main()