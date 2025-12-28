import lance
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
from tqdm import tqdm
import os
import argparse
import gc

# Default batch size
BATCH_SIZE = 500

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
        img = Image.fromarray(arr, mode='L')
    else:
        img = Image.fromarray(arr)
    
    # Encode to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def process_row(row):
    """Process a single row - can be used for debugging"""
    try:
        masks = bytes_to_array(row['masks'], row['shape'], row['masks_dtype'])
        crop_margin = bytes_to_array(row['crop_with_margin'], row['crop_shape'], row['crop_with_margin_dtype'])
        crop_mask = bytes_to_array(row['crop_mask'], row['crop_shape'], row['crop_mask_dtype'])
        
        # Check if img_binary exists
        if 'img_binary' in row and row['img_binary'] is not None:
            img = bytes_to_array(row['img_binary'], row['shape'], 'uint8')
            img_b64 = array_to_base64(img)
        else:
            img_b64 = None
        
        return {
            'index': int(row['index']),
            'filename': row['filename'],
            'provider': row['provider'],
            'image_type': row['image_type'],
            'n': int(row['n']),
            'xy_ratio': float(row['xy_ratio']) if not np.isnan(row['xy_ratio']) else None,
            'run_folder': row['run_folder'],
            'folder': row.get('folder', 'Unknown'),
            'path': row['path'],
            'masks_img': array_to_base64(masks),
            'crop_img': array_to_base64(crop_margin),
            'crop_mask_img': array_to_base64(crop_mask),
            'img': img_b64
        }
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def main(lance_path, output_dir, batch_size):
    print(f"Loading Lance dataset from: {lance_path}")
    
    # Load Lance dataset
    dataset = lance.dataset(lance_path)
    
    # Get total count
    total_images = dataset.count_rows()
    num_batches = (total_images + batch_size - 1) // batch_size
    
    print(f"Found {total_images} images")
    print(f"Will create {num_batches} batch files with {batch_size} images each")
    print(f"Columns: {dataset.schema.names}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process using index-based slicing instead of iterator
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx + 1}/{num_batches}")
        print(f"Rows {start_idx} to {end_idx-1} ({end_idx - start_idx} images)")
        print(f"{'='*60}")
        
        try:
            # Load this batch using take with indices
            print("Loading batch from Lance...")
            indices = list(range(start_idx, end_idx))
            batch_table = dataset.take(indices)
            
            print("Converting to pandas...")
            batch_df = batch_table.to_pandas()
            
            print(f"Processing {len(batch_df)} images...")
            web_data = []
            
            # Process with progress bar
            for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_idx + 1}"):
                result = process_row(row)
                if result is not None:
                    web_data.append(result)
            
            # Save batch to JSON
            output_file = os.path.join(output_dir, f"batch_{batch_idx + 1:03d}.json")
            print(f"Saving {len(web_data)} images to {output_file}...")
            
            with open(output_file, 'w') as f:
                json.dump(web_data, f)
            
            # Print file size
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"✓ Saved: {output_file} ({file_size_mb:.2f} MB)")
            
            # Clear memory
            del batch_table, batch_df, web_data
            gc.collect()
            
        except Exception as e:
            print(f"ERROR processing batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"DONE! Created {num_batches} batch files in '{output_dir}/' directory")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Load one batch at a time in the annotation tool")
    print(f"2. Annotate the batch")
    print(f"3. Export results")
    print(f"4. Load next batch")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Lance dataset to JSON batches for web annotation."
    )
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to the .lance dataset"
    )
    parser.add_argument(
        "--output", "-o", 
        default="annotation_batches", 
        help="Output directory for batch JSON files (default: annotation_batches)"
    )
    parser.add_argument(
        "--batch_size", "-b", 
        type=int,
        default=1000, 
        help="Number of images per JSON file (default: 500)"
    )
    parser.add_argument(
        "--start_batch", "-s",
        type=int,
        default=0,
        help="Start from batch number (useful if script crashed, default: 0)"
    )
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Start from: batch {args.start_batch + 1}\n")
    
    main(args.input, args.output, args.batch_size)