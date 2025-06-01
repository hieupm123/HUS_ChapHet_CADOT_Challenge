import json
import os
from PIL import Image
from tqdm import tqdm

# --- 1. Định nghĩa đường dẫn ---
image_input_dir = "../data/CADOT_Dataset/train"
annotation_file_path = "../data/CADOT_Dataset/train/_annotations.coco.json"
output_base_folder = "../data/split_folder_all_classes"

os.makedirs(output_base_folder, exist_ok=True)
print(f"Thư mục đầu ra chính: '{output_base_folder}'")

# --- 3. Đọc file annotation COCO ---
print(f"Đang đọc file annotation: {annotation_file_path}")
with open(annotation_file_path, 'r') as f:
    coco_data = json.load(f)

# Tạo dictionary để tra cứu nhanh
# image_id -> image_filename
images_info = {img['id']: img for img in coco_data['images']}
# category_id -> category_name
categories_info = {cat['id']: cat['name'] for cat in coco_data['categories']}

print(f"Tổng số ảnh trong annotation: {len(images_info)}")
print(f"Tổng số categories: {len(categories_info)}")
print(f"Tổng số annotations (đối tượng): {len(coco_data['annotations'])}")

# Để tạo tên file duy nhất cho các đối tượng được cắt ra từ cùng một ảnh
object_counts_per_image = {}

# --- 4. Lặp qua từng annotation để tách đối tượng ---
print("\nBắt đầu tách đối tượng...")
for ann in tqdm(coco_data['annotations'], desc="Đang xử lý annotations"):
    image_id = ann['image_id']
    category_id = ann['category_id']
    bbox = ann['bbox']  # Định dạng COCO: [x_min, y_min, width, height]

    # Lấy thông tin ảnh và category
    image_metadata = images_info.get(image_id)
    category_name_original = categories_info.get(category_id) # Lấy tên gốc

    if not image_metadata:
        print(f"Cảnh báo: Không tìm thấy thông tin cho image_id {image_id}. Bỏ qua annotation này.")
        continue
    if not category_name_original:
        print(f"Cảnh báo: Không tìm thấy tên category cho category_id {category_id}. Bỏ qua annotation này.")
        continue

    # ---- THAY ĐỔI CHÍNH ----
    # Thay thế space bằng underscore trong category_name để dùng cho tên thư mục và file
    safe_category_name = category_name_original.replace(" ", "_")

    original_image_filename = image_metadata['file_name']
    original_image_path = os.path.join(image_input_dir, original_image_filename)

    # Tạo thư mục con cho category nếu chưa có (sử dụng safe_category_name)
    category_output_dir = os.path.join(output_base_folder, safe_category_name)
    os.makedirs(category_output_dir, exist_ok=True)

    try:
        with Image.open(original_image_path) as img:
            x, y, w, h = bbox
            left = x
            upper = y
            right = x + w
            lower = y + h
            if w <= 0 or h <= 0:
                print(f"Cảnh báo: Bounding box không hợp lệ (width={w}, height={h}) cho đối tượng trong ảnh '{original_image_filename}'. Bỏ qua.")
                continue
            cropped_image = img.crop((left, upper, right, lower))

            if cropped_image.width == 0 or cropped_image.height == 0:
                print(f"Cảnh báo: Đối tượng cắt ra từ ảnh '{original_image_filename}' có kích thước 0. Bỏ qua.")
                continue

            # ---- THAY ĐỔI CHÍNH ----
            # Thay thế space bằng underscore trong base_name (tên file gốc không có phần mở rộng)
            base_name_no_ext, _ = os.path.splitext(original_image_filename)
            safe_base_name = base_name_no_ext.replace(" ", "_")

            current_obj_count = object_counts_per_image.get(original_image_filename, 0) + 1 # Dùng original_image_filename làm key vẫn ổn
            object_counts_per_image[original_image_filename] = current_obj_count
            ann_id_str = str(int(ann['id'])) # Đảm bảo ann['id'] là số và chuyển thành chuỗi

            # Sử dụng safe_base_name và safe_category_name để tạo tên file
            cropped_filename = f"{safe_base_name}_ann{ann_id_str}_{safe_category_name}.png"
            output_path = os.path.join(category_output_dir, cropped_filename)
            cropped_image.save(output_path)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh gốc: {original_image_path}. Bỏ qua annotation này.")
        continue
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý ảnh {original_image_filename} cho ann_id {ann['id']}: {e}. Bỏ qua annotation này.")
        continue


print("\nHoàn tất việc tách đối tượng!")
print(f"Các đối tượng đã được lưu trong thư mục: '{os.path.abspath(output_base_folder)}'")