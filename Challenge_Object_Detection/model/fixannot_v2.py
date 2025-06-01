import json
from collections import defaultdict

def modify_coco_annotations(input_coco_file, output_coco_file):
    """
    Modifies COCO annotations according to specified rules:
    - Redistributes 'building' instances among target place categories.
    - Redistributes 'small vehicle' instances among target vehicle categories.
    """
    try:
        with open(input_coco_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {input_coco_file}")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: File {input_coco_file} không phải là định dạng JSON hợp lệ.")
        return

    # 1. Tạo mapping category_id <-> category_name
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    category_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}

    # 2. Xác định ID nguồn và ID đích
    building_id = category_name_to_id.get('building')
    small_vehicle_id = category_name_to_id.get('small vehicle')

    target_place_names = [
        "graveyard",
        "playground",
        "swimming pool",
        'building'
    ]
    target_place_ids = [category_name_to_id[name] for name in target_place_names if name in category_name_to_id]

    target_vehicle_names = ["large vehicle", "medium vehicle", 'small vehicle']
    target_vehicle_ids = [category_name_to_id[name] for name in target_vehicle_names if name in category_name_to_id]

    if building_id is None:
        print("Cảnh báo: Không tìm thấy category 'building' trong file.")
    if small_vehicle_id is None:
        print("Cảnh báo: Không tìm thấy category 'small vehicle' trong file.")
    if not target_place_ids:
        print("Cảnh báo: Không tìm thấy category nào trong danh sách đích cho 'building'.")
    if not target_vehicle_ids:
        print("Cảnh báo: Không tìm thấy category nào trong danh sách đích cho 'small vehicle'.")

    # 3. Xử lý annotations
    new_annotations = []
    building_target_idx_counter = 0
    small_vehicle_target_idx_counter = 0

    for ann in data['annotations']:
        current_category_id = ann['category_id']
        modified_ann = ann.copy() # Tạo bản sao để chỉnh sửa

        if building_id is not None and current_category_id == building_id and target_place_ids:
            # Thay đổi category_id của 'building'
            new_cat_id = target_place_ids[building_target_idx_counter % len(target_place_ids)]
            modified_ann['category_id'] = new_cat_id
            building_target_idx_counter += 1
        elif small_vehicle_id is not None and current_category_id == small_vehicle_id and target_vehicle_ids:
            # Thay đổi category_id của 'small vehicle'
            new_cat_id = target_vehicle_ids[small_vehicle_target_idx_counter % len(target_vehicle_ids)]
            modified_ann['category_id'] = new_cat_id
            small_vehicle_target_idx_counter += 1
        
        new_annotations.append(modified_ann)

    # Cập nhật danh sách annotations trong data
    data['annotations'] = new_annotations

    # 4. Lưu file mới
    try:
        with open(output_coco_file, 'w') as f:
            json.dump(data, f, indent=4) # indent=4 để file JSON dễ đọc hơn
        print(f"Đã tạo file annotation mới: {output_coco_file}")
    except IOError:
        print(f"Lỗi: Không thể ghi file vào {output_coco_file}")
        return

    # 5. (Tùy chọn) In thống kê mới
    print("\n--- Số lượng mẫu mỗi nhãn sau khi chỉnh sửa ---")
    category_counts = defaultdict(int)
    valid_annotations_count = 0
    for ann in data['annotations']:
        if 'category_id' in ann:
            cat_name = category_id_to_name.get(ann['category_id'], f"unknown_id_{ann['category_id']}")
            category_counts[cat_name] += 1
            valid_annotations_count +=1

    # Lấy danh sách tất cả các category name từ data['categories'] để đảm bảo in cả các category có 0 mẫu
    all_category_names_defined = [cat['name'] for cat in data['categories']]
    
    for cat_name in sorted(all_category_names_defined):
        # Chỉ in những category còn lại sau khi đã loại bỏ "building" và "small vehicle" nếu chúng đã được thay thế hết
        # hoặc in tất cả nếu bạn muốn xem số lượng 0 cho "building" và "small vehicle"
        if cat_name not in ['building', 'small vehicle'] or category_counts[cat_name] > 0 :
             print(f"- {cat_name}: {category_counts[cat_name]}")
        elif cat_name in ['building', 'small vehicle'] and category_counts[cat_name] == 0: # In nếu chúng đã được thay thế hết
             print(f"- {cat_name} (đã được phân phối): {category_counts[cat_name]}")


    print("\n--- Tóm tắt sau khi chỉnh sửa ---")
    print(f"Tổng số chú thích đã xử lý (có category_id): {valid_annotations_count}")
    print(f"Tổng số lớp (categories) được định nghĩa: {len(data['categories'])}")
    print(f"Tổng số mẫu (instance) đã đếm: {sum(category_counts.values())}")


# --- Sử dụng ---
input_file_1 = "../data/CADOT_Dataset/train/_annotations.coco.json"
# Đặt tên file output, ví dụ: thêm "_modified" vào tên file gốc
output_file_parts_1 = input_file_1.split('.')
if len(output_file_parts_1) > 2 and output_file_parts_1[-2] == "coco": # Xử lý trường hợp .coco.json
    output_file_1 = ".".join(output_file_parts_1[:-2]) + "_modified.coco.json"
else: # Xử lý trường hợp .json
    output_file_1 = ".".join(output_file_parts_1[:-1]) + "_modified.json"




# Kiểm tra xem file input có tồn tại không trước khi chạy
# (Trong thực tế, bạn có thể thêm kiểm tra này hoặc để hàm tự xử lý)
# import os
# if not os.path.exists(input_file):
#     print(f"LỖI: File đầu vào '{input_file}' không tồn tại!")
# else:
modify_coco_annotations(input_file_1, output_file_1)

print("\n--- Các nhãn (categories) trong file mới (không thay đổi so với file gốc) ---")
# Để in lại danh sách categories nếu cần
# with open(output_file, 'r') as f:
#     updated_data = json.load(f)
# print(updated_data['categories'])