import json
import os
from ultralytics import YOLO
import torch
import re

# --- Cấu hình CHUNG ---
# Thư mục chứa ảnh gốc, có thể giữ cố định nếu tất cả cấu hình dùng chung
IMAGE_DIRECTORY = "../../data/CADOT_Dataset/test"

# --- DANH SÁCH CẤU HÌNH ---
# Mỗi mục trong danh sách này là một dict đại diện cho một lần chạy thử nghiệm
CONFIGURATIONS = [
    {
        "name": "infer_yolov11_origin", # Tên để dễ nhận diện
        "model_path": "../train/CADOT_YOLOv11_Origin/train_yolov11_01/weights/best.pt",
        "images_info_json_path": "image_ids.json", # File JSON chứa danh sách ảnh cho cấu hình này
        "confidence_threshold": 0.02,
        "iou_threshold": 0.6,
        "image_target_size": 512,
        "output_directory": "../../data/infer_results/infer_yolov11_origin", # THƯ MỤC LƯU TRỮ KẾT QUẢ
        "output_filename_prefix": "detections" # Tiền tố cho tên file kết quả
    },
    {
        "name": "infer_yolov11_v3",
        "model_path": "../train/CADOT_YOLOv11_v3/train_yolov11_01/weights/best.pt",
        "images_info_json_path": "image_ids.json",
        "confidence_threshold": 0.02,
        "iou_threshold": 0.69,
        "image_target_size": 512,
        "output_directory": "../../data/infer_results/infer_yolov11_v3",
        "output_filename_prefix": "detections"
    },
    {
        "name": "infer_yolov11_v5",
        "model_path": "../train/CADOT_YOLOv11_v5/train_yolov11_01/weights/best.pt",
        "images_info_json_path": "image_ids.json",
        "confidence_threshold": 0.01,
        "iou_threshold": 0.6,
        "image_target_size": 512,
        "output_directory": "../../data/infer_results/infer_yolov11_v5",
        "output_filename_prefix": "detections"
    },
    {
        "name": "infer_yolov12_v3",
        "model_path": "../train/CADOT_YOLOv12_v3/train_yolov12_01/weights/best.pt",
        "images_info_json_path": "image_ids.json",
        "confidence_threshold": 0.01,
        "iou_threshold": 0.5,
        "image_target_size": 512,
        "output_directory": "../../data/infer_results/infer_yolov12_v3",
        "output_filename_prefix": "detections"
    },
    {
        "name": "infer_yolov12_v5",
        "model_path": "../train/CADOT_YOLOv12_v5/train_yolov12_01/weights/best.pt",
        "images_info_json_path": "image_ids.json",
        "confidence_threshold": 0.01,
        "iou_threshold": 0.6,
        "image_target_size": 512,
        "output_directory": "../../data/infer_results/infer_yolov12_v5",
        "output_filename_prefix": "detections"
    },
    # Thêm các cấu hình khác vào đây
]

# --- Các hàm tiện ích ---
def sanitize_filename(name):
    """
    Thay thế các ký tự không hợp lệ trong tên file bằng dấu gạch dưới.
    """
    return re.sub(r'[^\w\.-]', '_', name)

# --- Hàm xử lý cho một cấu hình ---
def process_configuration(config, device, loaded_models_cache, loaded_image_metadata_cache):
    """
    Xử lý một cấu hình cụ thể từ danh sách CONFIGURATIONS.
    Sử dụng cache để tránh tải lại model và file JSON nếu đã được sử dụng trước đó.
    """
    config_name = config.get("name", "Unnamed Configuration")
    print(f"\n--- Starting processing for configuration: {config_name} ---")

    MODEL_PATH = config["model_path"]
    IMAGES_INFO_JSON_PATH = config["images_info_json_path"]
    CONFIDENCE_THRESHOLD = config["confidence_threshold"]
    IOU_THRESHOLD = config["iou_threshold"]
    IMAGE_TARGET_SIZE = config["image_target_size"]
    OUTPUT_DIRECTORY = config["output_directory"] # Đường dẫn đến thư mục lưu trữ kết quả
    OUTPUT_FILENAME_PREFIX = config["output_filename_prefix"] # Tiền tố tên file

    # 1. Tải Model (sử dụng cache)
    if MODEL_PATH in loaded_models_cache:
        model = loaded_models_cache[MODEL_PATH]
        print(f"Using cached model: '{MODEL_PATH}'")
    else:
        try:
            print(f"Loading model: '{MODEL_PATH}'...")
            model = YOLO(MODEL_PATH)
            model.to(device)
            loaded_models_cache[MODEL_PATH] = model
            print(f"Model '{MODEL_PATH}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{MODEL_PATH}': {e}")
            return # Bỏ qua cấu hình này nếu không tải được model

    class_names = model.names
    if not class_names:
        print("Warning: Could not retrieve class names from the model. Output files will use class IDs.")

    # 2. Tải thông tin ảnh (sử dụng cache)
    images_metadata_list = []
    if IMAGES_INFO_JSON_PATH in loaded_image_metadata_cache:
        images_metadata_list = loaded_image_metadata_cache[IMAGES_INFO_JSON_PATH]
        print(f"Using cached image metadata from: '{IMAGES_INFO_JSON_PATH}' ({len(images_metadata_list)} entries)")
    else:
        try:
            print(f"Loading image metadata from: '{IMAGES_INFO_JSON_PATH}'...")
            with open(IMAGES_INFO_JSON_PATH, 'r') as f:
                images_data_root = json.load(f)
            if "images" not in images_data_root:
                print(f"Error: Key 'images' not found in '{IMAGES_INFO_JSON_PATH}'. Expected format: {{'images': [...]}}")
                return
            images_metadata_list = images_data_root["images"]
            loaded_image_metadata_cache[IMAGES_INFO_JSON_PATH] = images_metadata_list
            print(f"Loaded {len(images_metadata_list)} image metadata entries from '{IMAGES_INFO_JSON_PATH}'.")
        except FileNotFoundError:
            print(f"Error: Images info JSON file not found at '{IMAGES_INFO_JSON_PATH}'.")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{IMAGES_INFO_JSON_PATH}'.")
            return
        except TypeError: # Handle case where JSON root is not a dict
            print(f"Error: Expected '{IMAGES_INFO_JSON_PATH}' to contain a JSON object with an 'images' key, but got {type(images_data_root)}.")
            return
        except Exception as e:
            print(f"An unexpected error occurred while loading image metadata from '{IMAGES_INFO_JSON_PATH}': {e}")
            return

    # Dictionary để lưu trữ kết quả theo class_id cho cấu hình hiện tại
    results_by_class = {}
    total_images_processed_config = 0
    total_detections_config = 0

    for image_info in images_metadata_list:
        image_id = image_info.get("id")
        file_name = image_info.get("file_name")

        if image_id is None or file_name is None:
            print(f"Warning: Skipping metadata entry due to missing 'id' or 'file_name': {image_info}")
            continue

        image_path = os.path.join(IMAGE_DIRECTORY, file_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}. Skipping.")
            continue

        total_images_processed_config += 1
        try:
            predictions = model.predict(
                image_path,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_TARGET_SIZE,
                verbose=False
            )

            if not predictions or len(predictions) == 0:
                continue

            result = predictions[0]
            boxes = result.boxes

            if len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])

                x_min, y_min, x_max, y_max = xyxy
                width = x_max - x_min
                height = y_max - y_min
                coco_bbox = [round(x_min, 2), round(y_min, 2), round(width, 2), round(height, 2)] # Round to 2 decimal places

                detection_data = {
                    "image_id": image_id,
                    "category_id": class_id, # This is the model's internal class ID
                    "bbox": coco_bbox,
                    "score": round(confidence, 4) # Round score
                }

                if class_id not in results_by_class:
                    results_by_class[class_id] = []
                results_by_class[class_id].append(detection_data)
                total_detections_config +=1

        except Exception as e:
            print(f"Error processing image {image_path} for config '{config_name}': {e}")
            continue
    
    print(f"Finished processing {total_images_processed_config} images for config '{config_name}'. Found {total_detections_config} total detections.")

    # Ghi kết quả ra nhiều file cho cấu hình hiện tại
    if not results_by_class:
        print(f"No detections found for any class in configuration '{config_name}'.")
    else:
        print(f"Saving COCO-style detection results for configuration '{config_name}':")
        
        # Tạo thư mục output_directory nếu chưa tồn tại
        if OUTPUT_DIRECTORY and not os.path.exists(OUTPUT_DIRECTORY):
            try:
                os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
                print(f"  Created output directory: {OUTPUT_DIRECTORY}")
            except OSError as e:
                print(f"Error creating directory {OUTPUT_DIRECTORY}: {e}. Results for this config may not be saved.")
                return # Không thể lưu file nếu không tạo được thư mục

        config_detections_count = 0
        for class_id, detections_for_class in results_by_class.items():
            if not detections_for_class:
                continue

            class_name_str = class_names.get(class_id, f"class_id_{class_id}") if class_names else f"class_id_{class_id}"
            safe_class_name = sanitize_filename(class_name_str)
            
            # Tạo tên file duy nhất dựa trên OUTPUT_FILENAME_PREFIX và tên class, lưu trong OUTPUT_DIRECTORY
            output_filename = f"{OUTPUT_FILENAME_PREFIX}_class_{safe_class_name}.json"
            output_file_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

            try:
                with open(output_file_path, 'w') as f:
                    json.dump(detections_for_class, f, indent=2)
                print(f"  - Results for class '{class_name_str}' (ID: {class_id}) with {len(detections_for_class)} detections saved to '{output_file_path}'")
                config_detections_count += len(detections_for_class)
            except IOError as e:
                print(f"Error: Could not write output JSON to '{output_file_path}': {e}")
        
        print(f"Total detections saved for configuration '{config_name}' into directory '{OUTPUT_DIRECTORY}': {config_detections_count}")
    print(f"--- Finished configuration: {config_name} ---")


# --- Hàm Main ---
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Global settings: Using device: {device}, Image Directory: {IMAGE_DIRECTORY}")

    if not CONFIGURATIONS:
        print("No configurations defined. Exiting.")
        return

    # Cache để lưu trữ các model và metadata đã tải
    loaded_models_cache = {}
    loaded_image_metadata_cache = {}

    # Tạo file image_ids.json mẫu nếu nó không tồn tại trong thư mục hiện tại của script
    # Điều này hữu ích nếu script được chạy từ một vị trí khác IMAGE_DIRECTORY
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_image_ids_json_path = os.path.join(script_dir, "image_ids.json")

    if not os.path.exists(default_image_ids_json_path):
        print(f"Attempting to create a sample 'image_ids.json' in '{script_dir}' as it was not found.")
        print("Please populate this file with the actual image IDs and filenames you wish to process.")
        print("The format should be: {\"images\": [{\"id\": 1, \"file_name\": \"image1.jpg\"}, ...]}")
        
        sample_images_data = {"images": []}
        # Thử quét IMAGE_DIRECTORY để tạo danh sách file mẫu (nếu IMAGE_DIRECTORY tồn tại và không trống)
        if os.path.isdir(IMAGE_DIRECTORY):
            print(f"Scanning '{IMAGE_DIRECTORY}' for sample image files...")
            try:
                image_files = [f for f in os.listdir(IMAGE_DIRECTORY) if os.path.isfile(os.path.join(IMAGE_DIRECTORY, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
                if image_files:
                    print(f"Found {len(image_files)} potential image files. Adding first few as samples.")
                    for i, fname in enumerate(image_files[:5]): # Lấy tối đa 5 file mẫu
                        sample_images_data["images"].append({"id": i + 1, "file_name": fname})
                else:
                    print(f"No image files found in '{IMAGE_DIRECTORY}'. Adding placeholder data.")
                    sample_images_data["images"] = [
                        {"id": 1, "file_name": "example_image1.jpg"},
                        {"id": 2, "file_name": "example_image2.png"}
                    ]
            except Exception as e:
                print(f"Could not scan IMAGE_DIRECTORY: {e}. Adding placeholder data.")
                sample_images_data["images"] = [
                    {"id": 1, "file_name": "example_image1.jpg"},
                    {"id": 2, "file_name": "example_image2.png"}
                ]
        else:
            print(f"IMAGE_DIRECTORY '{IMAGE_DIRECTORY}' not found or not a directory. Adding placeholder data to sample 'image_ids.json'.")
            sample_images_data["images"] = [
                {"id": 1, "file_name": "example_image1.jpg"},
                {"id": 2, "file_name": "example_image2.png"}
            ]

        try:
            with open(default_image_ids_json_path, 'w') as f:
                json.dump(sample_images_data, f, indent=2)
            print(f"Created sample '{default_image_ids_json_path}'. PLEASE EDIT IT with your actual image data and paths if necessary.")
            print("Make sure the 'images_info_json_path' in CONFIGURATIONS points to the correct file.")
        except IOError as e:
            print(f"Error creating sample '{default_image_ids_json_path}': {e}")

    # Chạy xử lý cho từng cấu hình
    for config_item in CONFIGURATIONS:
        # Đảm bảo images_info_json_path là đường dẫn tuyệt đối hoặc tương đối từ script
        # Nếu nó chỉ là tên file (ví dụ "image_ids.json"), giả sử nó nằm cùng thư mục script
        if not os.path.isabs(config_item["images_info_json_path"]) and not os.path.dirname(config_item["images_info_json_path"]):
            config_item["images_info_json_path"] = os.path.join(script_dir, config_item["images_info_json_path"])
            
        process_configuration(config_item, device, loaded_models_cache, loaded_image_metadata_cache)

    print("\nAll configurations processed.")

if __name__ == '__main__':
    main()