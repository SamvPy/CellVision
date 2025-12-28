import lance
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
from tqdm import tqdm
import os
import argparse

# Your Lance dataset path
BATCH_SIZE = 500  # Images per file - adjust based on your needs

def bytes_to_array(bytes_data, shape, dtype_str):
    """Reconstruct numpy array from bytes"""
    dtype = np.dtype(dtype_str)
    return np.frombuffer(bytes_data, dtype=dtype).reshape(shape)

def array_to_base64(arr):
    """Convert numpy array to base64 PNG for web display"""
    # Handle different dtypes
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        # Normalize to 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    
    # Convert to PIL Image
    if len(arr.shape) == 2:
        # Grayscale
        img = Image.fromarray(arr, mode='L')
    else:
        # RGB
        img = Image.fromarray(arr)
    
    # Encode to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def main(LANCE_PATH, OUTPUT_DIR, BATCH_SIZE):
    print(f"Loading Lance dataset from: {LANCE_PATH}")
    
    # Load Lance dataset
    dataset = lance.dataset(LANCE_PATH)
    batches = dataset.to_batches(batch_size=BATCH_SIZE)
    
    total_images = dataset.count_rows()
    num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Found {total_images} images")
    print(f"Will create {num_batches} batch files")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    batch_idx=0
    # Process in batches
    for batch in tqdm(batches, total=num_batches, desc="Processing batches"):
        
        batch_df = batch.to_pandas()
        web_data = []
        for idx, row in batch_df.iterrows():
            try:
                # Reconstruct arrays from bytes
                masks = bytes_to_array(row['masks'], row['shape'], row['masks_dtype'])
                crop_margin = bytes_to_array(row['crop_with_margin'], row['crop_shape'], row['crop_with_margin_dtype'])
                crop_mask = bytes_to_array(row['crop_mask'], row['crop_shape'], row['crop_mask_dtype'])
                img = bytes_to_array(row['img_binary'], row['shape'], 'uint8')
                
                # Convert to base64
                web_data.append({
                    'index': int(row['index']),
                    'filename': row['filename'],
                    'provider': row['provider'],
                    'image_type': row['image_type'],
                    'n': int(row['n']),
                    'xy_ratio': float(row['xy_ratio']) if not np.isnan(row['xy_ratio']) else None,
                    'run_folder': row['run_folder'],
                    'path': row['path'],
                    'masks_img': array_to_base64(masks),
                    'crop_img': array_to_base64(crop_margin),
                    'crop_mask_img': array_to_base64(crop_mask),
                    'img': array_to_base64(img)
                })
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Save batch to JSON
        output_file = os.path.join(OUTPUT_DIR, f"batch_{batch_idx + 1:03d}.json")
        print(f"Saving {len(web_data)} images to {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(web_data, f)

        batch_idx += 1
        # Print file size
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
    
    print(f"\nDone! Created {num_batches} batch files in '{OUTPUT_DIR}/' directory")
    print(f"Load one batch at a time in the annotation tool.")
    print(f"\nRecommendation: Annotate one batch, export results, then load next batch.")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Convert Lance dataset to JSON for web annotation.")
    parser.add_argument("--input", "-i", required=True, help="Path to the .lance dataset")
    parser.add_argument("--output", "-o", default="dataset.json", help="Path to save the output JSON")
    parser.add_argument("--batch_size", "-b", default=500, help="Number of images per JSON output file.")
    args = parser.parse_args()

    lance_path = args.input
    output_json = args.output
    batch_size = args.batch_size
    main(lance_path, output_json, batch_size)