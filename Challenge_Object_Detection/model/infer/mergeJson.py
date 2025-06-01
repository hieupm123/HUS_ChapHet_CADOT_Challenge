import json
import os

def merge_json_files(list_of_input_files, output_file_path):
    """
    Gộp nội dung từ nhiều file JSON (giả định mỗi file chứa một danh sách)
    thành một file JSON duy nhất chứa một danh sách lớn.

    Args:
        list_of_input_files (list): Một danh sách các chuỗi, mỗi chuỗi là đường dẫn
                                    đến một file JSON đầu vào.
        output_file_path (str): Đường dẫn đến file JSON output đã gộp.
    """
    all_data_merged = []

    if not list_of_input_files:
        print("No input files provided for merging.")
        return

    print(f"Attempting to merge {len(list_of_input_files)} files:")
    for file_path in list_of_input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue

        print(f"  - Reading: {file_path}")
        try:
            with open(file_path, 'r') as f:
                data_from_file = json.load(f)

            # Giả định rằng nội dung chính của file là một danh sách
            # Nếu không, bạn có thể cần điều chỉnh logic này để trích xuất
            # phần danh sách mong muốn từ data_from_file
            if isinstance(data_from_file, list):
                all_data_merged.extend(data_from_file)
                print(f"    Successfully read and added {len(data_from_file)} items.")
            else:
                print(f"    Warning: Content of {file_path} is not a list. Type: {type(data_from_file)}. Skipping content integration as list.")

        except json.JSONDecodeError:
            print(f"    Error: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"    An unexpected error occurred while processing {file_path}: {e}. Skipping.")

    if not all_data_merged:
        print("No data was collected from the input files to merge.")
        return

    # Lưu danh sách tất cả dữ liệu đã gộp vào file output
    try:
        # Đảm bảo thư mục output tồn tại
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        with open(output_file_path, 'w') as f:
            json.dump(all_data_merged, f, indent=2)
        print(f"\nSuccessfully merged data from {len(list_of_input_files)} specified files into '{output_file_path}'.")
        print(f"Total items in merged file: {len(all_data_merged)}")
    except IOError:
        print(f"Error: Could not write merged data to '{output_file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output file: {e}.")

# --- Cấu hình cho việc gộp file ---

if __name__ == '__main__':
    # --- BẠN CẦN ĐỊNH NGHĨA DANH SÁCH FILE Ở ĐÂY ---
    # Thay thế các đường dẫn này bằng đường dẫn thực tế đến các file JSON bạn muốn gộp.
    # Các file này có thể ở bất kỳ đâu trên hệ thống tệp của bạn.
    files_to_merge = [
        "../../data/infer_results/infer_yolov12_v3/detections_class_basketball_field.json",
        "../../data/infer_results/infer_yolov12_v3/detections_class_building.json",
        "../../data/infer_results/infer_yolov11_origin/detections_class_crosswalk.json",
        "../../data/infer_results/infer_yolov11_v5/detections_class_football_field.json",        # Ví dụ từ câu hỏi trước
        "../../data/infer_results/infer_yolov12_v5/detections_class_graveyard.json",
        "../../data/infer_results/infer_yolov11_origin/detections_class_large_vehicle.json",
        "../../data/infer_results/infer_yolov12_v3/detections_class_medium_vehicle.json",
        "../../data/infer_results/infer_yolov11_v5/detections_class_playground.json",
        "../../data/infer_results/infer_yolov11_v3/detections_class_roundabout.json",
        "../../data/infer_results/infer_yolov11_v3/detections_class_ship.json",
        "../../data/infer_results/infer_yolov11_v3/detections_class_small_vehicle.json",
        "../../data/infer_results/infer_yolov11_origin/detections_class_swimming_pool.json",
        "../../data/infer_results/infer_yolov11_v3/detections_class_tennis_court.json",
        "../../data/infer_results/infer_yolov11_v5/detections_class_train.json"
    ]

    # Bạn có thể muốn đặt tên này một cách rõ ràng hơn
    MERGED_OUTPUT_JSON_PATH_CUSTOM = "../../results/Predictions.json"

    # Gọi hàm gộp
    merge_json_files(files_to_merge, MERGED_OUTPUT_JSON_PATH_CUSTOM)