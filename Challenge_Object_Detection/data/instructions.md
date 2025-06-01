## 0. `CADOT_Dataset`
   - **Contains:** Image source of contest.

## 1. `split_folder_all_classes`
   - **Contains:** Individually cropped object images from source, sorted by category.
   - **Structure:** `split_folder_all_classes/Category_Name/original_basename_ann<ID>_Category.png`
   - **Purpose:** Provides easy access to classified object instances for model training.

## 2. `outargument`
   - **Contains:** Rotated versions (data augmentation) of images from `split_folder_all_classes`.
   - **Structure:** `outargument/Category_Name/original_object_filename_rotSUFFIX.png`
   - **Purpose:** Increases dataset diversity with rotated image variations.

## 3. `outargument_color_variants`
   - **Contains:** Color-transformed versions of images from `outargument`.
   - **Structure:** `outargument_color_variants/Category_Name/rotated_filename_colorSUFFIX.png`
   - **Purpose:** Further enhances dataset diversity with color variations.

## 4. `out_data_with_new`
   - **Contains:** Cropped versions of images from `outargument_color_variants` (only the last crop type applied is saved).
   - **Structure:** `out_data_with_new/Category_Name/filename_from_color_variants.png`
   - **Purpose:** Generates specific cropped image variations (currently only one per input).

## 5. `train_inpainted_cv2`
   - **Contains:** Images where specified object classes have been removed via OpenCV inpainting, plus an updated COCO JSON.
   - **Structure:** `train_inpainted_cv2/image_name.ext` and `_annotations.coco.json`.
   - **Purpose:** Creates a dataset with certain object classes visually and anntationally removed.

## 6. `augmented_dataset_direct_paste_v1` (and `_v2` to `_v6` variants)
   - **Contains:** Original images augmented by "directly pasting" new object instances (of the same class) into existing bounding boxes, plus an updated COCO JSON.
   - **Structure:** `augmented_dataset_direct_paste_vX/image_name.ext` and `_annotations.coco.json`.
   - **Purpose:** Increases object instance variety within original scene contexts.

## 7. `new_train_v3`
   - **Contains:** A merged dataset from `augmented_dataset_direct_paste_v1` through `_v5`, and the original `CADOT_Dataset/train`, with all images and a unified COCO JSON.
   - **Structure:** `new_train_v3/data/suffixed_image_name.ext` and `_annotations.coco.json`.
   - **Purpose:** Consolidates multiple original and augmented datasets into one large training set.

## 8. `new_train_v5`
   - **Contains:** A merged dataset from `augmented_dataset_direct_paste_v6` and the original `CADOT_Dataset/train`, with all images and a unified COCO JSON.
   - **Structure:** `new_train_v5/data/suffixed_image_name.ext` and `_annotations.coco.json`.
   - **Purpose:** Combines original data with a specific augmented version (v6) into a unified training set.