import os
import json
import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import traceback # Để in traceback đầy đủ khi có lỗi

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Cấu hình ---
INPUT_IMAGE_FOLDER = "../data/CADOT_Dataset/train"
COCO_ANNOTATION_FILE = "../data/CADOT_Dataset/train/_annotations.coco.json"
OUTPUT_FOLDER = "../data/train_inpainted_cv2" # Đổi tên thư mục output
INPAINT_METHOD = cv2.INPAINT_NS # Hoặc cv2.INPAINT_NS
INPAINT_RADIUS = 5 # Bán kính vùng lân cận để inpainting, có thể cần điều chỉnh

# --- DANH SÁCH CÁC LỚP SẼ BỊ XÓA (INPAINT) VÀ LOẠI BỎ KHỎI ANNOTATION MỚI ---
CATEGORIES_TO_REMOVE_FOR_INPAINTING = [
    "large vehicle",
    "medium vehicle",
    "small vehicle",
    "crosswalk"
]
# ---------------------------------------------------------------------------

# --- Kiểm tra và tạo thư mục output nếu chưa có ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Đã tạo thư mục output: {OUTPUT_FOLDER}")

# --- Bước 1: Không cần khởi tạo mô hình AI ---
print(f"Sử dụng thuật toán OpenCV Inpainting: {'TELEA' if INPAINT_METHOD == cv2.INPAINT_TELEA else 'NAVIER_STOKES'}")

# --- Bước 2: Đọc file COCO và chuẩn bị dữ liệu ---
print(f"Đang đọc file COCO: {COCO_ANNOTATION_FILE}")
if not os.path.exists(COCO_ANNOTATION_FILE):
    print(f"Lỗi: Không tìm thấy file COCO: {COCO_ANNOTATION_FILE}")
    exit()

original_coco_data = None
try:
    with open(COCO_ANNOTATION_FILE, 'r') as f:
        original_coco_data = json.load(f)
except Exception as e:
    print(f"Lỗi khi đọc file JSON gốc {COCO_ANNOTATION_FILE}: {e}")
    traceback.print_exc()
    exit()

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask_utils
    coco = COCO(COCO_ANNOTATION_FILE)
except ImportError:
    print("Lỗi: Không thể import pycocotools. Hãy chắc chắn bạn đã cài đặt nó.")
    print("Chạy lệnh: pip install pycocotools")
    exit()
except Exception as e:
    print(f"Lỗi khi khởi tạo đối tượng COCO từ pycocotools: {e}")
    traceback.print_exc()
    exit()

category_name_to_id = {cat['name']: cat['id'] for cat in original_coco_data.get('categories', [])}
ids_to_remove = set()
for cat_name in CATEGORIES_TO_REMOVE_FOR_INPAINTING:
    if cat_name in category_name_to_id:
        ids_to_remove.add(category_name_to_id[cat_name])
    else:
        print(f"Cảnh báo: Không tìm thấy category '{cat_name}' trong file COCO.")
print(f"Các Category ID sẽ được xóa/inpaint (CV2) và loại bỏ khỏi COCO mới: {ids_to_remove}")


image_ids_from_coco_obj = coco.getImgIds()
images_info_list = coco.loadImgs(image_ids_from_coco_obj)
print(f"Tìm thấy {len(images_info_list)} ảnh trong file COCO.")

new_coco_global_annotations = []
successfully_processed_image_ids = set()

# --- Bước 3, 4, 5: Xử lý từng ảnh và thu thập annotations cho file mới ---
for img_info in tqdm(images_info_list, desc="Đang xử lý ảnh (CV2 Inpaint)"):
    image_filename = img_info['file_name']
    image_id = img_info['id']
    image_path = os.path.join(INPUT_IMAGE_FOLDER, image_filename)
    output_image_path = os.path.join(OUTPUT_FOLDER, image_filename)

    if not os.path.exists(image_path):
        print(f"Cảnh báo: Không tìm thấy file ảnh {image_path}. Bỏ qua ảnh này.")
        continue

    try:
        original_image_pil = Image.open(image_path).convert("RGB")
        original_image_np = np.array(original_image_pil) # original_image_np sẽ là dạng RGB (HxWx3)
        if original_image_np.ndim == 2: # Nếu ảnh grayscale (sau convert hiếm khi xảy ra)
             original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_GRAY2RGB)
        elif original_image_np.shape[2] == 4: # Nếu có kênh alpha
            original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_RGBA2RGB) # Hoặc original_image_np[:,:,:3]

    except Exception as e:
        print(f"Lỗi khi đọc hoặc xử lý sơ bộ ảnh {image_path}: {e}. Bỏ qua.")
        traceback.print_exc()
        continue

    current_image_ann_ids = coco.getAnnIds(imgIds=image_id)
    current_image_all_annotations = coco.loadAnns(current_image_ann_ids)

    mask_np = np.zeros(original_image_np.shape[:2], dtype=np.uint8) # Mask phải là 8-bit grayscale (HxW)
    has_objects_to_inpaint = False
    current_image_kept_annotations = []

    if not current_image_all_annotations:
        try:
            original_image_pil.save(output_image_path)
            successfully_processed_image_ids.add(image_id)
        except Exception as e:
            print(f"Lỗi khi lưu ảnh (không có annotations) {output_image_path}: {e}")
        continue

    for ann in current_image_all_annotations:
        if ann['category_id'] in ids_to_remove:
            has_objects_to_inpaint = True
            if 'segmentation' in ann:
                seg = ann['segmentation']
                if isinstance(seg, list) and seg:
                    for poly in seg:
                        if len(poly) >= 6:
                            poly_np_arr = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask_np, [poly_np_arr], 255)
                elif isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
                    rle = coco.annToRLE(ann)
                    decoded_mask_for_ann = coco_mask_utils.decode(rle)
                    if decoded_mask_for_ann.ndim == 3:
                        decoded_mask_for_ann = np.any(decoded_mask_for_ann, axis=2)
                    mask_np = np.maximum(mask_np, decoded_mask_for_ann.astype(np.uint8) * 255)
                elif 'bbox' in ann and not seg:
                    bbox = [int(c) for c in ann['bbox']]
                    x, y, w, h = bbox
                    cv2.rectangle(mask_np, (x, y), (x + w, y + h), 255, -1)
            elif 'bbox' in ann:
                bbox = [int(c) for c in ann['bbox']]
                x, y, w, h = bbox
                cv2.rectangle(mask_np, (x, y), (x + w, y + h), 255, -1)
        else:
            current_image_kept_annotations.append(ann)
    
    # Dilate mask một chút có thể cải thiện kết quả inpainting ở biên
    # if has_objects_to_inpaint:
    # kernel = np.ones((3,3),np.uint8) # Kernel nhỏ hơn có thể phù hợp hơn với CV2 inpaint
    # mask_np = cv2.dilate(mask_np, kernel, iterations = 1)


    try:
        if has_objects_to_inpaint:
            # original_image_np là uint8, 3 kênh (RGB)
            # mask_np là uint8, 1 kênh
            inpainted_result_np = cv2.inpaint(original_image_np, mask_np, INPAINT_RADIUS, INPAINT_METHOD)
            # Kết quả từ cv2.inpaint đã là np.ndarray (uint8)
            inpainted_image_to_save = Image.fromarray(inpainted_result_np)
            inpainted_image_to_save.save(output_image_path)
        else:
            original_image_pil.save(output_image_path)

        successfully_processed_image_ids.add(image_id)
        new_coco_global_annotations.extend(current_image_kept_annotations)

    except cv2.error as cve:
        print(f"\nLỗi OpenCV khi inpainting ảnh {image_filename}: {cve}")
        print(f"  Loại ảnh đầu vào cho cv2.inpaint: {type(original_image_np)}, shape: {original_image_np.shape}, dtype: {original_image_np.dtype}")
        print(f"  Loại mask đầu vào cho cv2.inpaint: {type(mask_np)}, shape: {mask_np.shape}, dtype: {mask_np.dtype}")
        traceback.print_exc()
        print(f"Bỏ qua ảnh {image_filename} do lỗi OpenCV.")
        continue
    except Exception as e:
        print(f"\nLỗi không xác định khi inpainting (CV2) hoặc lưu ảnh {image_filename}: {e}")
        traceback.print_exc()
        print(f"Bỏ qua ảnh {image_filename} và các annotations của nó.")
        continue

print("\nHoàn thành xử lý tất cả các ảnh!")

# --- Bước 6: Tạo file COCO annotation mới ---
print("\nĐang tạo file COCO annotation mới...")

new_coco_data_images = []
for img_data_from_list in images_info_list:
    if img_data_from_list['id'] in successfully_processed_image_ids:
        new_coco_data_images.append(img_data_from_list)

new_coco_data_images.sort(key=lambda img: img['id'])
new_coco_global_annotations.sort(key=lambda ann: (ann['image_id'], ann['id']))

final_new_coco_dict = {
    'info': original_coco_data.get('info', {}),
    'licenses': original_coco_data.get('licenses', []),
    'categories': original_coco_data.get('categories', []),
    'images': new_coco_data_images,
    'annotations': new_coco_global_annotations
}

output_coco_path = os.path.join(OUTPUT_FOLDER, "_annotations.coco.json")
try:
    with open(output_coco_path, 'w') as f:
        json.dump(final_new_coco_dict, f, indent=4)
    print(f"Đã tạo file COCO annotation mới: {output_coco_path}")

    print("\n--- Thống kê sơ bộ file COCO mới ---")
    print(f"Số lượng ảnh: {len(final_new_coco_dict['images'])}")
    print(f"Số lượng chú thích: {len(final_new_coco_dict['annotations'])}")
    category_counts_new = {}
    for cat in final_new_coco_dict['categories']:
        category_counts_new[cat['id']] = {'name': cat['name'], 'count': 0}
    for ann in final_new_coco_dict['annotations']:
        if ann['category_id'] in category_counts_new:
            category_counts_new[ann['category_id']]['count'] += 1
    
    print("Số lượng mẫu mỗi nhãn trong file mới:")
    for cat_id, data in category_counts_new.items():
        if data['count'] > 0 or category_name_to_id.get(data['name']) not in ids_to_remove:
            print(f"- {data['name']} (id: {cat_id}): {data['count']}")
        elif category_name_to_id.get(data['name']) in ids_to_remove:
             print(f"- {data['name']} (id: {cat_id}): {data['count']} (đã được lọc bỏ instances)")

except Exception as e:
    print(f"Lỗi khi ghi file COCO JSON mới: {e}")
    traceback.print_exc()

print("\n--- Xong! ---")