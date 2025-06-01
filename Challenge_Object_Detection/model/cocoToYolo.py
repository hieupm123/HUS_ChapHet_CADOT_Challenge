import json
import os
from tqdm import tqdm # Thư viện để hiển thị thanh tiến trình (cài đặt: pip install tqdm)

# !!! QUAN TRỌNG: Danh sách TÊN các lớp đối tượng theo ĐÚNG THỨ TỰ index (từ 0) mà bạn muốn sử dụng.
# Thứ tự này sẽ quyết định ID lớp trong file YOLO.
PREDEFINED_GLOBAL_CLASS_NAMES = [
    'small-object', 'basketball field', 'building', 'crosswalk', 'football field', 
    'graveyard', 'large vehicle', 'medium vehicle', 'playground', 'roundabout', 
    'ship', 'small vehicle', 'swimming pool', 'tennis court', 'train'
]

# Thay thế bằng các đường dẫn thực tế của bạn
datasets_to_process = [
    {
        "coco_annotation_file": '../data/new_train_v3/_annotations.coco.json',
        "image_directory": '../data/new_train_v3/data'
        # Các file .txt sẽ được lưu vào chính thư mục image_directory được chỉ định ở đây
    },
    {
        "coco_annotation_file": '../data/new_train_v5/_annotations.coco.json',
        "image_directory": '../data/new_train_v5/data'
        # Các file .txt sẽ được lưu vào chính thư mục image_directory được chỉ định ở đây
    },
    {
        "coco_annotation_file": '../data/CADOT_Dataset/train/_annotations.coco.json',
        "image_directory": '../data/CADOT_Dataset/train'
        # Các file .txt sẽ được lưu vào chính thư mục image_directory được chỉ định ở đây
    },
    {
        "coco_annotation_file": '../data/CADOT_Dataset/valid/_annotations.coco.json',
        "image_directory": '../data/CADOT_Dataset/valid'
        # Các file .txt sẽ được lưu vào chính thư mục image_directory được chỉ định ở đây
    },
]


# --- Hàm chuyển đổi tọa độ COCO sang YOLO ---
def coco_to_yolo(bbox, img_width, img_height):
    """
    Chuyển đổi bounding box từ format COCO [x_min, y_min, width, height]
    sang format YOLO [x_center_norm, y_center_norm, width_norm, height_norm].
    """
    x_min, y_min, w, h = bbox
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    return x_center_norm, y_center_norm, width_norm, height_norm

# --- Hàm tạo mapping lớp toàn cục ---
def create_global_class_mapping(all_coco_files, predefined_class_names_ordered):
    """
    Sử dụng danh sách lớp (class names) đã định nghĩa trước và thứ tự của chúng
    để tạo mapping từ tên lớp sang YOLO ID (0-based).
    Đồng thời, quét qua các file COCO để kiểm tra xem có lớp nào trong file COCO
    mà không có trong danh sách định nghĩa trước không.
    """
    print("--- Bước 0: Tạo Mapping Lớp Toàn Cục từ Danh Sách Định Nghĩa Trước ---")

    if not predefined_class_names_ordered:
        print("LỖI NGHIÊM TRỌNG: Danh sách lớp định nghĩa trước (PREDEFINED_GLOBAL_CLASS_NAMES) bị trống!")
        return None, None
        
    if len(set(predefined_class_names_ordered)) != len(predefined_class_names_ordered):
        print("CẢNH BÁO NGHIÊM TRỌNG: Danh sách lớp định nghĩa trước (PREDEFINED_GLOBAL_CLASS_NAMES) chứa tên lớp bị trùng lặp. "
              "Điều này sẽ dẫn đến việc gán ID lớp không chính xác. Vui lòng sửa lại danh sách.")
        # Có thể quyết định dừng hẳn ở đây: raise ValueError("...")
        # Hoặc return None, None để dừng ở main_conversion_orchestrator

    # Sử dụng trực tiếp danh sách và thứ tự đã định nghĩa trước
    global_yolo_class_names = list(predefined_class_names_ordered) 
    global_class_name_to_yolo_id = {name: i for i, name in enumerate(global_yolo_class_names)}

    print("\nMapping Tên Lớp Toàn Cục sang YOLO ID (0-based) DỰA TRÊN DANH SÁCH ĐỊNH NGHĨA TRƯỚC:")
    for name, yolo_id in global_class_name_to_yolo_id.items():
        print(f"  '{name}' -> {yolo_id}")
    print(f"Tổng số lớp định nghĩa (nc): {len(global_yolo_class_names)}")

    # Bây giờ, quét các file COCO để kiểm tra các lớp có trong đó
    print("\n--- Kiểm tra các lớp trong file COCO so với danh sách định nghĩa trước ---")
    all_coco_category_names_found_in_files = set()
    found_categories_in_at_least_one_file = False

    for coco_file_path in all_coco_files:
        print(f"  Đang quét file: {coco_file_path}")
        try:
            with open(coco_file_path, 'r') as f:
                coco_data = json.load(f)
            categories = coco_data.get('categories', [])
            if not categories:
                print(f"  THÔNG TIN: Không tìm thấy 'categories' trong {coco_file_path}. Bỏ qua file này cho việc kiểm tra lớp.")
                continue
            
            found_categories_in_at_least_one_file = True
            for category in categories:
                coco_class_name = category['name']
                all_coco_category_names_found_in_files.add(coco_class_name)
                if coco_class_name not in global_class_name_to_yolo_id:
                    print(f"  CẢNH BÁO QUAN TRỌNG: Lớp '{coco_class_name}' tìm thấy trong {coco_file_path} "
                          f"KHÔNG có trong danh sách lớp toàn cục định nghĩa trước (PREDEFINED_GLOBAL_CLASS_NAMES). "
                          f"Các annotation cho lớp này sẽ bị BỎ QUA trong quá trình chuyển đổi.")
        except FileNotFoundError:
            print(f"  LỖI: Không tìm thấy file annotation: {coco_file_path}")
            continue # Tiếp tục với file khác
        except json.JSONDecodeError:
            print(f"  LỖI: File annotation không phải JSON hợp lệ: {coco_file_path}")
            continue
        except Exception as e:
            print(f"  LỖI không xác định khi đọc file {coco_file_path}: {e}")
            continue
            
    if not found_categories_in_at_least_one_file and all_coco_files:
        print("CẢNH BÁO: Không tìm thấy mục 'categories' trong bất kỳ file COCO nào được cung cấp để kiểm tra tên lớp.")
    
    # Kiểm tra ngược lại: Lớp nào trong predefined list không xuất hiện trong bất kỳ COCO file nào
    predefined_set = set(global_yolo_class_names)
    classes_in_predefined_not_in_any_coco = predefined_set - all_coco_category_names_found_in_files
    if classes_in_predefined_not_in_any_coco:
        print("\n  THÔNG TIN: Các lớp sau có trong danh sách PREDEFINED_GLOBAL_CLASS_NAMES nhưng "
              "KHÔNG tìm thấy trong bất kỳ trường 'categories' của file COCO nào đã quét:")
        for name in sorted(list(classes_in_predefined_not_in_any_coco)):
            print(f"    - '{name}'")
    else:
        if found_categories_in_at_least_one_file : #Chỉ in nếu đã thực sự quét được category nào đó
             print("\n  THÔNG TIN: Tất cả các lớp trong PREDEFINED_GLOBAL_CLASS_NAMES (nếu có trong COCO files) đều được tìm thấy.")
    
    print("--- Hoàn thành tạo Mapping Lớp Toàn Cục và Kiểm Tra ---\n")
    
    return global_class_name_to_yolo_id, global_yolo_class_names

# --- Hàm chính để xử lý chuyển đổi cho một bộ dữ liệu ---
def process_single_dataset(coco_annotation_file, image_directory, global_class_name_to_yolo_id, global_yolo_class_names_ordered_list):
    print(f"\n--- Đang xử lý bộ dữ liệu ---")
    print(f"File COCO: {coco_annotation_file}")
    print(f"Thư mục ảnh (và lưu .txt): {image_directory}")

    try:
        with open(coco_annotation_file, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file annotation tại: {coco_annotation_file}. Bỏ qua bộ dữ liệu này.")
        return 0
    except json.JSONDecodeError:
        print(f"LỖI: File annotation không phải là định dạng JSON hợp lệ: {coco_annotation_file}. Bỏ qua bộ dữ liệu này.")
        return 0
    except Exception as e:
        print(f"LỖI không xác định khi đọc file JSON {coco_annotation_file}: {e}. Bỏ qua bộ dữ liệu này.")
        return 0

    # --- Bước 1 (Cục bộ): Tạo mapping cần thiết cho BỘ DỮ LIỆU HIỆN TẠI ---
    # Mapping từ category_id (COCO cục bộ) sang YOLO ID (toàn cục)
    
    local_coco_cat_id_to_global_yolo_id = {}
    categories_from_coco = coco_data.get('categories', [])
    if not categories_from_coco:
        print(f"CẢNH BÁO: Không tìm thấy phần 'categories' trong file {coco_annotation_file}. Các annotation (nếu có) sẽ không thể map class ID và có thể bị bỏ qua.")
    else:
        for category_info in categories_from_coco:
            coco_local_id = category_info['id']
            class_name = category_info['name']
            if class_name in global_class_name_to_yolo_id:
                local_coco_cat_id_to_global_yolo_id[coco_local_id] = global_class_name_to_yolo_id[class_name]
            else:
                # Cảnh báo này đã được đưa ra ở `create_global_class_mapping`, nhưng để ở đây cũng tốt để biết cụ thể file nào bị.
                print(f"  LƯU Ý (trong process_single_dataset): Lớp '{class_name}' (ID COCO cục bộ: {coco_local_id}) trong file {coco_annotation_file} "
                      f"không nằm trong danh sách PREDEFINED_GLOBAL_CLASS_NAMES. "
                      f"Các annotation cho lớp này sẽ bị bỏ qua.")

    # Mapping từ image_id sang thông tin ảnh (tên file, kích thước)
    image_id_to_info = {img['id']: img for img in coco_data.get('images', [])}
    if not image_id_to_info:
        print(f"LỖI: Không tìm thấy phần 'images' trong file {coco_annotation_file}. Bỏ qua bộ dữ liệu này.")
        return 0

    # Gom annotations theo image_id để xử lý nhanh hơn
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    print(f"Tìm thấy {len(image_id_to_info)} ảnh và {len(coco_data.get('annotations', []))} annotations trong {coco_annotation_file}.")

    # --- Bước 2: Duyệt qua từng ảnh và tạo file .txt ---
    print(f"Bắt đầu tạo file label YOLO trong thư mục: {image_directory}")
    if not os.path.exists(image_directory):
        try:
            os.makedirs(image_directory, exist_ok=True)
            print(f"Đã tạo thư mục: {image_directory}")
        except OSError as e:
            print(f"LỖI: Không thể tạo thư mục {image_directory}: {e}. Bỏ qua bộ dữ liệu này.")
            return 0
    elif not os.path.isdir(image_directory):
        print(f"LỖI: Đường dẫn {image_directory} tồn tại nhưng không phải là thư mục. Bỏ qua bộ dữ liệu này.")
        return 0


    processed_files_count = 0
    images_in_json = list(image_id_to_info.values())

    for img_info in tqdm(images_in_json, desc=f"Processing images for {os.path.basename(coco_annotation_file)}"):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        
        # Xử lý trường hợp file_name có thể chứa đường dẫn con
        img_basename = os.path.basename(img_filename)
        img_width = img_info['width']
        img_height = img_info['height']

        base_filename_no_ext = os.path.splitext(img_basename)[0]
        label_filename = f"{base_filename_no_ext}.txt"
        label_filepath = os.path.join(image_directory, label_filename)

        img_annotations = annotations_by_image.get(img_id, [])

        with open(label_filepath, 'w') as f_label:
            for ann in img_annotations:
                coco_local_cat_id = ann['category_id']
                bbox = ann['bbox']

                if coco_local_cat_id not in local_coco_cat_id_to_global_yolo_id:
                    # Đã cảnh báo ở trên khi tạo local_coco_cat_id_to_global_yolo_id nếu class_name không có trong global list.
                    # Chỉ bỏ qua nếu ID cục bộ không được map.
                    # print(f"\nCẢNH BÁO: Bỏ qua annotation có category_id cục bộ ({coco_local_cat_id}) không map được "
                    #       f"với YOLO ID toàn cục, trong ảnh {img_filename} của file {coco_annotation_file}")
                    continue
                
                yolo_global_class_id = local_coco_cat_id_to_global_yolo_id[coco_local_cat_id]

                try:
                    if img_width <= 0 or img_height <= 0: # Kiểm tra kỹ hơn, không chỉ bằng 0
                        print(f"\nLỖI: Ảnh {img_filename} (ID: {img_id}) có width hoặc height không hợp lệ ({img_width}x{img_height}). "
                              f"Không thể chuẩn hóa. Bỏ qua annotation này.")
                        continue 

                    x_center_n, y_center_n, w_n, h_n = coco_to_yolo(bbox, img_width, img_height)

                    # Kiểm tra tọa độ chuẩn hóa. Đôi khi bbox COCO có thể nằm ngoài lề ảnh một chút.
                    # YOLO thường kỳ vọng tọa độ nằm trong [0, 1]. 
                    # Cân nhắc việc clamp giá trị hoặc bỏ qua. Hiện tại đang bỏ qua.
                    if not (0 <= x_center_n <= 1 and 0 <= y_center_n <= 1 and 0 <= w_n <= 1 and 0 <= h_n <= 1):
                         # print(f"\nCẢNH BÁO: Tọa độ chuẩn hóa không hợp lệ cho bbox {bbox} trong ảnh {img_filename} (kích thước {img_width}x{img_height}). Bỏ qua annotation này.")
                         # print(f"   Giá trị chuẩn hóa: x={x_center_n:.4f}, y={y_center_n:.4f}, w={w_n:.4f}, h={h_n:.4f}")
                         # Ví dụ về clamping (tùy chọn):
                         # x_center_n = max(0, min(1, x_center_n))
                         # y_center_n = max(0, min(1, y_center_n))
                         # w_n = max(0, min(1, w_n))
                         # h_n = max(0, min(1, h_n))
                         # Và sau đó kiểm tra if w_n > 0 and h_n > 0
                         continue

                    f_label.write(f"{yolo_global_class_id} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}\n")

                except ZeroDivisionError: 
                     print(f"\nLỖI (ZeroDivisionError): Ảnh {img_filename} có width hoặc height bằng 0 sau khi kiểm tra ban đầu. Bỏ qua annotation.")
                     continue
                except Exception as e:
                    print(f"\nLỖI khi xử lý annotation {ann.get('id', 'N/A')} cho ảnh {img_filename}: {e}")
        
        processed_files_count += 1
    
    print(f"Đã xử lý và tạo/cập nhật file .txt cho {processed_files_count} ảnh từ {coco_annotation_file}.")
    return processed_files_count

# --- Hàm chính để điều phối toàn bộ quá trình ---
def main_conversion_orchestrator():
    if not datasets_to_process:
        print("Không có bộ dữ liệu nào được cấu hình trong 'datasets_to_process'. Kết thúc.")
        return

    # Lấy danh sách tất cả các file COCO để kiểm tra với mapping lớp toàn cục
    all_coco_files = [d["coco_annotation_file"] for d in datasets_to_process]
    
    # Tạo mapping lớp toàn cục DỰA TRÊN DANH SÁCH ĐỊNH NGHĨA TRƯỚC
    global_class_name_to_yolo_id, global_yolo_class_names_ordered = create_global_class_mapping(all_coco_files, PREDEFINED_GLOBAL_CLASS_NAMES)

    if global_class_name_to_yolo_id is None or global_yolo_class_names_ordered is None:
        print("LỖI: Không thể tạo mapping lớp toàn cục (có thể do danh sách PREDEFINED_GLOBAL_CLASS_NAMES trống hoặc có lỗi). Dừng chương trình.")
        return
    
    if not global_yolo_class_names_ordered: # Sẽ được bắt bởi kiểm tra ở trên, nhưng để chắc chắn
        print("CẢNH BÁO: Không có lớp nào trong danh sách định nghĩa. Các file YOLO có thể sẽ trống hoặc không được tạo đúng cách.")
        # return # Có thể muốn dừng hẳn ở đây

    total_files_processed_across_all_datasets = 0
    for dataset_info in datasets_to_process:
        coco_file = dataset_info["coco_annotation_file"]
        img_dir = dataset_info["image_directory"]
        
        count = process_single_dataset(coco_file, img_dir, global_class_name_to_yolo_id, global_yolo_class_names_ordered)
        total_files_processed_across_all_datasets += count

    print(f"\n--- Hoàn thành chuyển đổi cho TẤT CẢ các bộ dữ liệu ---")
    print(f"Tổng cộng đã xử lý và tạo/cập nhật file .txt cho {total_files_processed_across_all_datasets} ảnh.")
    
    print("\nThông tin lớp TOÀN CỤC (từ PREDEFINED_GLOBAL_CLASS_NAMES) để sử dụng trong file dataset.yaml của bạn:")
    print(f"nc: {len(global_yolo_class_names_ordered)}")
    print(f"names: {global_yolo_class_names_ordered}")
    print("\nĐảm bảo file dataset.yaml của bạn sử dụng đúng 'nc' và 'names' này, đặc biệt nếu bạn kết hợp các bộ dữ liệu này để huấn luyện.")
    print("LƯU Ý: Các annotation cho lớp không có trong PREDEFINED_GLOBAL_CLASS_NAMES sẽ bị bỏ qua.")

if __name__ == "__main__":
    main_conversion_orchestrator()