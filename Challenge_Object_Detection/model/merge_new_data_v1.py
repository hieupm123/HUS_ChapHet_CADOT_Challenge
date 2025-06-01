import json
import os
import shutil
from tqdm import tqdm # For progress bars, install with: pip install tqdm

def merge_coco_datasets(dataset_infos, output_folder, output_json_name="_annotations.coco.json"):
    """
    Merges multiple COCO datasets into a single one.

    Args:
        dataset_infos (list): A list of dictionaries, where each dictionary contains:
            'path': Path to the dataset folder (containing images and the COCO JSON).
            'json_name': Name of the COCO JSON file in that folder.
            'suffix': Suffix to add to image filenames from this dataset.
        output_folder (str): Path to the output folder for the merged dataset.
        output_json_name (str): Name for the merged COCO JSON file.
    """
    # Create output directories
    output_image_dir = os.path.join(output_folder, "data") # Convention for COCO is often images in 'data' or 'images' subfolder
    os.makedirs(output_image_dir, exist_ok=True)

    merged_coco = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # --- Global counters and mappings for merging ---
    next_image_id = 1
    next_annotation_id = 1
    
    # Category handling: map category names to new IDs to ensure uniqueness
    # and consistency across datasets.
    category_name_to_new_id = {}
    next_category_id = 1 
    
    processed_at_least_one_dataset = False

    for ds_info in dataset_infos:
        dataset_path = ds_info['path']
        json_file_name = ds_info['json_name']
        image_suffix = ds_info['suffix']
        
        coco_json_path = os.path.join(dataset_path, json_file_name)

        if not os.path.exists(coco_json_path):
            print(f"Warning: COCO JSON file not found at {coco_json_path}. Skipping this dataset.")
            continue

        print(f"Processing dataset: {dataset_path} with suffix '{image_suffix}'")
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        if not processed_at_least_one_dataset:
            # Copy info and licenses from the first dataset
            merged_coco["info"] = coco_data.get("info", {})
            merged_coco["licenses"] = coco_data.get("licenses", [])
            processed_at_least_one_dataset = True
        
        # --- 1. Process Categories ---
        # Maps old category IDs from *this specific dataset* to new global category IDs
        current_dataset_old_cat_id_to_new_cat_id = {} 
        
        for category in coco_data.get("categories", []):
            cat_name = category["name"]
            old_cat_id = category["id"]
            
            if cat_name not in category_name_to_new_id:
                category_name_to_new_id[cat_name] = next_category_id
                new_cat_entry = category.copy() # Copy all fields like 'supercategory'
                new_cat_entry["id"] = next_category_id
                merged_coco["categories"].append(new_cat_entry)
                current_dataset_old_cat_id_to_new_cat_id[old_cat_id] = next_category_id
                next_category_id += 1
            else:
                # Category name already exists, map to its existing new ID
                current_dataset_old_cat_id_to_new_cat_id[old_cat_id] = category_name_to_new_id[cat_name]

        # --- 2. Process Images and copy them ---
        # Maps old image IDs from *this specific dataset* to new global image IDs
        current_dataset_old_image_id_to_new_image_id = {}
        
        print(f"  Copying images and updating image entries for {os.path.basename(dataset_path)}...")
        for image_info in tqdm(coco_data.get("images", []), desc="Images"):
            old_image_id = image_info["id"]
            original_file_name = image_info["file_name"]
            
            base_name, ext = os.path.splitext(original_file_name)
            new_file_name = f"{base_name}{image_suffix}{ext}"
            
            src_image_path = os.path.join(dataset_path, original_file_name) # Assuming images are in the root of dataset_path
            # If images are in a subfolder like 'images' within dataset_path, adjust here:
            # src_image_path = os.path.join(dataset_path, 'images', original_file_name) 
            dst_image_path = os.path.join(output_image_dir, new_file_name)
            
            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)
            else:
                print(f"Warning: Image file not found: {src_image_path}. Skipping this image entry.")
                continue

            new_image_entry = image_info.copy() # Keep all original image metadata
            new_image_entry["id"] = next_image_id
            new_image_entry["file_name"] = new_file_name
            
            merged_coco["images"].append(new_image_entry)
            current_dataset_old_image_id_to_new_image_id[old_image_id] = next_image_id
            next_image_id += 1

        # --- 3. Process Annotations ---
        print(f"  Updating annotation entries for {os.path.basename(dataset_path)}...")
        for ann_info in tqdm(coco_data.get("annotations", []), desc="Annotations"):
            old_image_id = ann_info["image_id"]
            old_category_id = ann_info["category_id"]

            # Only process annotations for images that were successfully copied/found
            if old_image_id not in current_dataset_old_image_id_to_new_image_id:
                # This can happen if an image was listed in JSON but not found on disk
                # or if the image had no annotations (but then this loop wouldn't run for it).
                # print(f"Warning: Skipping annotation for old_image_id {old_image_id} as image was not processed.")
                continue
            
            if old_category_id not in current_dataset_old_cat_id_to_new_cat_id:
                print(f"Warning: Skipping annotation due to unknown old_category_id {old_category_id}. "
                      f"Image ID: {old_image_id}, Ann ID: {ann_info['id']}")
                continue

            new_ann_entry = ann_info.copy() # Keep all original annotation metadata
            new_ann_entry["id"] = next_annotation_id
            new_ann_entry["image_id"] = current_dataset_old_image_id_to_new_image_id[old_image_id]
            new_ann_entry["category_id"] = current_dataset_old_cat_id_to_new_cat_id[old_category_id]
            
            merged_coco["annotations"].append(new_ann_entry)
            next_annotation_id += 1
            
    # --- Remove duplicate categories by (id, name) pair after merging ---
    # This step ensures categories from different files but with same name get the same ID
    # The current category processing logic already handles this by mapping names to unique new IDs.
    # So, `merged_coco["categories"]` should already be correct and de-duplicated by name.

    # --- Save the merged COCO JSON ---
    output_json_path = os.path.join(output_folder, output_json_name)
    print(f"\nSaving merged COCO annotations to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(merged_coco, f, indent=4) # indent for readability

    print("Merging complete!")
    print(f"Total images: {len(merged_coco['images'])}")
    print(f"Total annotations: {len(merged_coco['annotations'])}")
    print(f"Total categories: {len(merged_coco['categories'])}")


if __name__ == "__main__":
    # --- Configuration ---

    # Define your datasets
    datasets_to_merge = [
        {
            "path": "../data/augmented_dataset_direct_paste_v1",
            "json_name": "_annotations.coco.json",
            "suffix": "_v1"
        },
        {
            "path": "../data/augmented_dataset_direct_paste_v2",
            "json_name": "_annotations.coco.json",
            "suffix": "_v2"
        },
        {
            "path": "../data/augmented_dataset_direct_paste_v3",
            "json_name": "_annotations.coco.json",
            "suffix": "_v3"
        },
        {
            "path": "../data/augmented_dataset_direct_paste_v4",
            "json_name": "_annotations.coco.json",
            "suffix": "_v4"
        },
        {
            "path": "../data/augmented_dataset_direct_paste_v5",
            "json_name": "_annotations.coco.json",
            "suffix": "_v5"
        },
        {
            "path": "../data/CADOT_Dataset/train",
            "json_name": "_annotations.coco.json",
            "suffix": "_origin"
        }
        # Add more datasets here if needed
    ]

    output_folder_name = "../data/new_train_v3"
    # full_output_path = os.path.join(output_base_dir, output_folder_name)

    # --- Run the merging process ---
    merge_coco_datasets(datasets_to_merge, output_folder_name, "_annotations.coco.json")