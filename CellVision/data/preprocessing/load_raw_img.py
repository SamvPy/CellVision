from glob import glob
from PIL import Image
import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def collect_paths(root, folder, run_folder_level):
    paths = []
    for i in glob(f'{root}/{folder}/**/*.png', recursive=True):
    
        paths.append(
            {
                'dir': os.path.dirname(i),
                'path': i,
                'folder': i.split('/')[6],
                'run_folder': '/'.join(i.split('/')[:run_folder_level]),
                'filename': i.split('/')[-1],
            }
        )
    return paths

def get_run_folder(path, default):
    folder_names = path[1:].split('/')
    run_folder = ''
    for folder_name in folder_names:
        run_folder += '/'+folder_name
        if run_folder.endswith('.Run') or run_folder.endswith('.Mapping') or run_folder.endswith('.Analysis'):
            return run_folder
    return default

def add_shapes(df, path_column='path'):
    df['shape'] = df[path_column].progress_apply(
        lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE).shape
    )

    df['img_height'] = df['shape'].apply(lambda x: x[1])
    df['img_width'] = df['shape'].apply(lambda x: x[0])

def annotate_image_type(row):
    if 'background' in row['filename'].lower():
        return 'background'
    if 'blue' in row['filename'].lower():
        return 'Fluorescent(blue)'
    if 'red' in row['filename'].lower():
        return 'Fluorescent(red)'
    if 'green' in row['filename'].lower():
        return 'Fluorescent(green)'
    return 'Brightfield_cell'
    
def get_img_dataframe(root, folder_name, shapes=True, img_type=True):
    paths = pd.DataFrame(collect_paths(root, folder_name, 8))
    paths['provider'] = folder_name
    paths['run_folder'] = paths.apply(
        lambda x: get_run_folder(path=x['path'], default=x['run_folder']), axis=1
    )
    if shapes:
        add_shapes(paths)
    if img_type:
        paths['image_type'] = paths.apply(annotate_image_type, axis=1)
    return paths