import json
import os
import cv2
import random
import copy
import shutil
import numpy as np

# --- Configuration ---
ORIGINAL_IMAGE_DIR = "../data/CADOT_Dataset/train"
COCO_ANNOTATION_FILE = "../data/CADOT_Dataset/train/_annotations_modified.coco.json"

OBJECT_SOURCE_DIRS = {
    # "ship": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/outargument_color_variants/ship",
    # "train": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/outargument_color_variants/train",
    # "basketball field": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/out_data_with_new/basketball_fields",
    # "crosswalk": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/split_folder_all_classes/crosswalk",
    # "football field": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/out_data_with_new/football_field",
    "graveyard": "../data/out_data_with_new/graveyard",
    "large vehicle": "../data/outargument_color_variants/large_vehicle",
    "medium vehicle": "../data/split_folder_all_classes/medium_vehicle",
    "playground": "../data/out_data_with_new/playground",
    # "roundabout": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/out_data_with_new/roundabout",
    "swimming pool": "../data/out_data_with_new/swimming_pool"
    # "tennis court": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/out_data_with_new/tennis_court",
    # "building": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/split_folder_all_classes/building",
    # "small vehicle": "/datausers3/kttv/tien/ClassificationProjectHimawari/test/split_folder_all_classes/small_vehicle"
}

OUTPUT_BASE_DIR = "../data/augmented_dataset_direct_paste_v6" 
OUTPUT_AUGMENTED_IMAGE_DIR = "../data/augmented_dataset_direct_paste_v6" 
OUTPUT_AUGMENTED_ANNOTATION_FILE = os.path.join(OUTPUT_BASE_DIR, "_annotations.coco.json")

# --- Helper Functions ---
def get_image_annotations(coco_data, image_id):
    return [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

def get_category_maps(coco_data):
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    return category_id_to_name, category_name_to_id

# --- Main Logic ---
def main():
    print("Starting augmentation process with Direct Paste (v1)...")

    if os.path.exists(OUTPUT_BASE_DIR):
        print(f"Output directory {OUTPUT_BASE_DIR} already exists. Clearing it.")
        shutil.rmtree(OUTPUT_BASE_DIR)
    os.makedirs(OUTPUT_AUGMENTED_IMAGE_DIR, exist_ok=True)
    print(f"Created output image directory: {OUTPUT_AUGMENTED_IMAGE_DIR}")

    print(f"Loading COCO annotations from: {COCO_ANNOTATION_FILE}")
    with open(COCO_ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)

    category_id_to_name, _ = get_category_maps(coco_data)

    new_coco_data = {
        'info': copy.deepcopy(coco_data.get('info', {})),
        'licenses': copy.deepcopy(coco_data.get('licenses', [])),
        'categories': copy.deepcopy(coco_data['categories']),
        'images': [],
        'annotations': []
    }
    
    global_new_annotation_id = 0 # Initialize new annotation ID counter

    total_images = len(coco_data['images'])
    print(f"Processing {total_images} images...")

    for i, img_info_orig in enumerate(coco_data['images']):
        original_image_id = img_info_orig['id']
        original_file_name = img_info_orig['file_name']
        original_image_path = os.path.join(ORIGINAL_IMAGE_DIR, original_file_name)

        if (i + 1) % 50 == 0 or (i + 1) == total_images:
            print(f"  Processing image {i+1}/{total_images}: {original_file_name}")

        if not os.path.exists(original_image_path):
            print(f"    Warning: Original image not found: {original_image_path}. Skipping image.")
            continue

        base_image = cv2.imread(original_image_path)
        if base_image is None:
            print(f"    Warning: Could not read image: {original_image_path}. Skipping image.")
            continue
        
        augmented_image = base_image.copy()
        img_h_orig, img_w_orig = augmented_image.shape[:2]
        
        # Ensure augmented_image (base_image) is 3-channel for pasting
        if len(augmented_image.shape) == 2 or augmented_image.shape[2] == 1: # Grayscale
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_GRAY2BGR)
        elif augmented_image.shape[2] == 4: # BGRA
            augmented_image = augmented_image[:,:,:3] # Drop alpha

        current_image_annotations_orig = get_image_annotations(coco_data, original_image_id)
        new_annotations_for_this_image = []

        for ann_orig in current_image_annotations_orig:
            category_id = ann_orig['category_id']
            category_name = category_id_to_name.get(category_id)
            
            # --- Prepare for potential paste operation ---
            attempt_paste = False # Flag to control if we proceed with paste steps
            current_ann_pixels_modified = False # Flag if pixels for this annotation were changed

            if category_name and category_name in OBJECT_SOURCE_DIRS:
                attempt_paste = True # Eligible for pasting, proceed with checks

            # Bbox for current annotation
            bbox_x, bbox_y, bbox_w, bbox_h = [int(v) for v in ann_orig['bbox']]

            if attempt_paste and (bbox_w <= 0 or bbox_h <= 0):
                # print(f"      Skipping invalid bbox [w,h <=0]: {[bbox_x, bbox_y, bbox_w, bbox_h]} for {category_name} in {original_file_name}")
                attempt_paste = False

            if attempt_paste:
                replacement_source_dir = OBJECT_SOURCE_DIRS[category_name]
                if not os.path.exists(replacement_source_dir) or not os.listdir(replacement_source_dir):
                    # print(f"      Warning: Replacement source dir for '{category_name}' is empty or not found. Keeping original.")
                    attempt_paste = False
            
            if attempt_paste:
                replacement_files = [
                    f for f in os.listdir(replacement_source_dir) 
                    if os.path.isfile(os.path.join(replacement_source_dir, f))
                       and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ]
                if not replacement_files:
                    # print(f"      Warning: No suitable image files found in {replacement_source_dir} for {category_name}. Keeping original.")
                    attempt_paste = False

            if attempt_paste:
                chosen_replacement_file = random.choice(replacement_files)
                replacement_image_path = os.path.join(replacement_source_dir, chosen_replacement_file)
                replacement_obj_img_raw = cv2.imread(replacement_image_path, cv2.IMREAD_UNCHANGED)
                if replacement_obj_img_raw is None:
                    # print(f"      Warning: Could not read replacement object {replacement_image_path}. Skipping.")
                    attempt_paste = False

            if attempt_paste:
                obj_mask = None # Will be single-channel binary mask
                replacement_obj_bgr = None # Will be 3-channel BGR

                if len(replacement_obj_img_raw.shape) == 3 and replacement_obj_img_raw.shape[2] == 4: # RGBA
                    replacement_obj_bgr = replacement_obj_img_raw[:, :, :3]
                    alpha_channel = replacement_obj_img_raw[:, :, 3]
                    _, obj_mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
                elif len(replacement_obj_img_raw.shape) == 3 and replacement_obj_img_raw.shape[2] == 3: # RGB/BGR
                    replacement_obj_bgr = replacement_obj_img_raw
                    obj_mask = 255 * np.ones(replacement_obj_bgr.shape[:2], dtype=np.uint8) # Full mask
                elif len(replacement_obj_img_raw.shape) == 2: # Grayscale
                    replacement_obj_bgr = cv2.cvtColor(replacement_obj_img_raw, cv2.COLOR_GRAY2BGR)
                    obj_mask = 255 * np.ones(replacement_obj_bgr.shape[:2], dtype=np.uint8) # Full mask
                else:
                    # print(f"      Warning: Unsupported image format for {chosen_replacement_file}. Skipping.")
                    attempt_paste = False
                
                if replacement_obj_bgr is None or obj_mask is None: # Should be caught by above, but double check
                    attempt_paste = False

            if attempt_paste:
                try:
                    resized_replacement_obj = cv2.resize(replacement_obj_bgr, (bbox_w, bbox_h), interpolation=cv2.INTER_AREA)
                    resized_mask = cv2.resize(obj_mask, (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
                except cv2.error as e:
                    # print(f"      Error resizing replacement obj or mask: {e}. Skipping paste for this object.")
                    attempt_paste = False
                
                # Check if resized images have valid dimensions (already checked bbox_w, bbox_h > 0)
                if attempt_paste and (resized_replacement_obj.shape[0] == 0 or resized_replacement_obj.shape[1] == 0):
                    # print(f"      Resized replacement object has zero dimension. Skipping paste.")
                    attempt_paste = False

            # --- Perform Direct Paste if all checks passed ---
            if attempt_paste:
                # Define target RoI using annotation bbox coordinates
                roi_y_start, roi_y_end = bbox_y, bbox_y + bbox_h
                roi_x_start, roi_x_end = bbox_x, bbox_x + bbox_w

                # Boundary checks for pasting RoI into augmented_image
                if not (0 <= roi_y_start < roi_y_end <= img_h_orig and \
                        0 <= roi_x_start < roi_x_end <= img_w_orig):
                    # print(f"      RoI for paste [{bbox_x},{bbox_y},{bbox_w},{bbox_h}] is outside image bounds. Skipping paste.")
                    pass # attempt_paste remains True but paste won't happen
                else:
                    try:
                        # Extract background RoI from the (potentially already modified) augmented image
                        bg_roi = augmented_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                        # Ensure bg_roi is 3-channel (augmented_image should already be)
                        if len(bg_roi.shape) < 3 or bg_roi.shape[2] != 3:
                             raise ValueError("Background RoI is not 3-channel BGR for pasting.")

                        # Apply mask to the foreground object (resized_replacement_obj is BGR)
                        masked_fg = cv2.bitwise_and(resized_replacement_obj, resized_replacement_obj, mask=resized_mask)
                        
                        # Create inverse mask for background
                        inv_mask = cv2.bitwise_not(resized_mask)
                        
                        # Apply inverse mask to background RoI
                        masked_bg = cv2.bitwise_and(bg_roi, bg_roi, mask=inv_mask)
                        
                        # Combine masked foreground and masked background
                        combined_roi = cv2.add(masked_fg, masked_bg)
                        
                        # Place the combined RoI back into the augmented image
                        augmented_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = combined_roi
                        current_ann_pixels_modified = True # Indicate that pixels were changed
                    
                    except cv2.error as e_cv:
                        print(f"      OpenCV error during direct paste for ann_id={ann_orig['id']} (img: {original_file_name}): {e_cv}. Keeping original.")
                    except Exception as e_gen:
                        print(f"      Generic error during direct paste for ann_id={ann_orig['id']} (img: {original_file_name}): {e_gen}. Keeping original.")
            
            # --- Add annotation to the list for this image ---
            # Create a new annotation object by copying the original one.
            # Assign a new unique ID and ensure 'image_id' points to the current (new) image.
            new_ann_for_output = copy.deepcopy(ann_orig)
            new_ann_for_output['id'] = global_new_annotation_id
            new_ann_for_output['image_id'] = original_image_id # This image's ID in new_coco_data

            new_annotations_for_this_image.append(new_ann_for_output)
            global_new_annotation_id += 1
            # current_ann_pixels_modified flag can be used for logging if needed

        # Save the (potentially) augmented image
        output_image_filepath = os.path.join(OUTPUT_AUGMENTED_IMAGE_DIR, original_file_name)
        cv2.imwrite(output_image_filepath, augmented_image)

        # Add new image info to new_coco_data
        new_img_info = copy.deepcopy(img_info_orig) # id here is original_image_id
        new_coco_data['images'].append(new_img_info)
        
        # Add all processed annotations for this image to the global list
        new_coco_data['annotations'].extend(new_annotations_for_this_image)

    # Save new COCO data
    print(f"Saving augmented COCO annotations to: {OUTPUT_AUGMENTED_ANNOTATION_FILE}")
    with open(OUTPUT_AUGMENTED_ANNOTATION_FILE, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

    print("Augmentation process with Direct Paste (v1) completed.")
    print(f"Augmented images saved in: {OUTPUT_AUGMENTED_IMAGE_DIR}")
    print(f"New annotations saved in: {OUTPUT_AUGMENTED_ANNOTATION_FILE}")

if __name__ == '__main__':
    main()