from ultralytics import YOLO
import torch
import os

# --- Cấu hình chung ---
DATASET_YAML_PATH = 'Cadot_detect_object_origin.yaml'
# THAY ĐỔI CHÍNH Ở ĐÂY:
BASE_MODEL_NAME = 'yolo11l.pt' # Giả sử tên model mới là yolov11n.pt
# Các lựa chọn có thể là: yolov11n.pt, yolov11s.pt, yolov11m.pt, yolov11l.pt, yolov11x.pt (nếu tồn tại)
PROJECT_NAME = 'CADOT_YOLOv11_Origin' # Có thể đổi tên dự án để phân biệt
EXPERIMENT_NAME_TRAIN = 'train_yolov11_01' # Tên cụ thể cho lần huấn luyện này
EXPERIMENT_NAME_VAL = 'val_after_yolov11_train_01'  
EXPERIMENT_NAME_PREDICT = 'predict_yolov11_run_01'

# GHI CHÚ QUAN TRỌNG:
# Nếu "YOLOv11" là một lớp Python riêng biệt trong thư viện, bạn cần:
# 1. Thay đổi câu lệnh import, ví dụ:
#    from ultralytics.models.yolov11 import YOLOv11 # (Đây là ví dụ, đường dẫn thực tế có thể khác)
# 2. Thay đổi cách khởi tạo model trong các hàm bên dưới, ví dụ:
#    model = YOLOv11(BASE_MODEL_NAME) # thay vì YOLO(BASE_MODEL_NAME)
#    model = YOLOv11(checkpoint_path) # thay vì YOLO(checkpoint_path)
# Tuy nhiên, với cách Ultralytics thường làm, khả năng cao là bạn vẫn dùng lớp `YOLO` chung.

def write_log(message, filename='log_yolov11.txt', mode='a'): # Đổi tên file log nếu muốn
    """
    Ghi một dòng log vào file.

    Parameters:
        message (str): Nội dung cần ghi log.
        filename (str): Tên file log. Mặc định là 'log_yolov11.txt'.
        mode (str): Chế độ ghi file, mặc định là 'a' (append). Dùng 'w' để ghi đè.
    """
    with open(filename, mode, encoding='utf-8') as f:
        f.write(message + '\n')

# Kiểm tra và thiết lập GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Đang sử dụng thiết bị: {device}")
if device == 'cuda':
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    device_id = 0 # Sử dụng GPU đầu tiên
else:
    device_id = 'cpu'

def train_model():
    """Huấn luyện một model YOLO mới (đã được cập nhật cho YOLOv11)."""
    print(f"--- Bắt đầu Huấn luyện Model ({BASE_MODEL_NAME}) ---")

    # Tải một model YOLOv11 được huấn luyện trước (ví dụ: yolov11n.pt)
    # Hoặc bạn có thể tải một checkpoint đã huấn luyện trước đó
    model = YOLO(BASE_MODEL_NAME) # Sử dụng lớp YOLO gốc, chỉ định model qua tên

    # Huấn luyện model
    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=300,
        imgsz=512,
        batch=16,
        device=device_id,
        name=EXPERIMENT_NAME_TRAIN,
        project=PROJECT_NAME,
        patience=20,
        save_period=10,
    )
    print("--- Huấn luyện Hoàn tất ---")
    print(f"Kết quả huấn luyện được lưu tại: {results.save_dir}")
    print(f"Model tốt nhất được lưu tại: {os.path.join(results.save_dir, 'weights/best.pt')}")
    return os.path.join(results.save_dir, 'weights/best.pt') # Trả về đường dẫn best.pt

def resume_training(checkpoint_path):
    """Tiếp tục huấn luyện từ một checkpoint (đã được cập nhật cho YOLOv11)."""
    print(f"--- Tiếp tục Huấn luyện từ Checkpoint: {checkpoint_path} ---")
    if not os.path.exists(checkpoint_path):
        print(f"Lỗi: Checkpoint '{checkpoint_path}' không tồn tại.")
        return None

    model = YOLO(checkpoint_path) # Sử dụng lớp YOLO gốc

    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=300,
        imgsz=512,
        batch=16, # Cân nhắc giảm nếu VRAM không đủ cho YOLOv11 lớn hơn
        device=device_id,
        name=EXPERIMENT_NAME_TRAIN + "_resumed",
        project=PROJECT_NAME,
        resume=True,
    )
    print("--- Tiếp tục Huấn luyện Hoàn tất ---")
    print(f"Kết quả huấn luyện được lưu tại: {results.save_dir}")
    return os.path.join(results.save_dir, 'weights/best.pt') # Trả về đường dẫn best.pt

def validate_model(model_weights_path):
    """Đánh giá model trên tập validation (đã được cập nhật cho YOLOv11)."""
    print(f"--- Bắt đầu Đánh giá Model: {model_weights_path} ---")
    if not os.path.exists(model_weights_path):
        print(f"Lỗi: File trọng số '{model_weights_path}' không tồn tại.")
        return

    model = YOLO(model_weights_path) # Sử dụng lớp YOLO gốc

    metrics = model.val(
        data=DATASET_YAML_PATH,
        imgsz=512,
        batch=16, # Cân nhắc giảm nếu VRAM không đủ
        device=device_id,
        name=EXPERIMENT_NAME_VAL,
        project=PROJECT_NAME,
        split='val',
    )
    print("--- Đánh giá Hoàn tất ---")
    print(f"Kết quả đánh giá được lưu tại: {metrics.save_dir}")
    print("Các chỉ số chính:")
    if hasattr(metrics, 'box') and hasattr(metrics.box, 'map'):
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP75: {metrics.box.map75:.4f}")
    else:
        print("  Không thể truy cập các chỉ số mAP. Vui lòng kiểm tra đối tượng metrics.")

def predict_with_model(model_weights_path, image_source):
    """Thực hiện dự đoán (đã được cập nhật cho YOLOv11)."""
    print(f"--- Bắt đầu Dự đoán với Model: {model_weights_path} ---")
    if not os.path.exists(model_weights_path):
        print(f"Lỗi: File trọng số '{model_weights_path}' không tồn tại.")
        return
    if isinstance(image_source, str) and not (os.path.exists(image_source) or image_source.startswith(('http://', 'https://'))):
        print(f"Lỗi: Nguồn ảnh/video '{image_source}' không tồn tại hoặc không phải URL.")
        return

    model = YOLO(model_weights_path) # Sử dụng lớp YOLO gốc

    results = model.predict(
        source=image_source,
        imgsz=512,
        conf=0.25,
        iou=0.45,
        device=device_id,
        save=True,
        save_txt=True,
        save_conf=True,
        name=EXPERIMENT_NAME_PREDICT,
        project=PROJECT_NAME,
    )
    print("--- Dự đoán Hoàn tất ---")

    for i, r in enumerate(results):
        if isinstance(image_source, str) and os.path.isdir(image_source) and hasattr(r, 'path'):
             # Nếu r.path là tên file trong thư mục, ghép với thư mục nguồn
             # Tuy nhiên, r.path thường đã là đường dẫn đầy đủ tới ảnh gốc
             current_image_path = r.path
        elif hasattr(r, 'path'):
             current_image_path = r.path
        else: # Fallback if r.path is not available
             current_image_path = f"Nguồn {i+1}"


        print(f"\nKết quả cho: {current_image_path}")
        if hasattr(r, 'save_dir'):
            print(f"  Kết quả đã lưu vào thư mục: {r.save_dir}")
        else:
            print(f"  Đường dẫn lưu kết quả không có sẵn trong đối tượng results.")


        boxes = r.boxes
        if boxes:
            print(f"  Tìm thấy {len(boxes)} đối tượng.")
            for box in boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id] if model.names else f"Lớp {class_id}"
                confidence = float(box.conf)
                xyxy = box.xyxy[0].cpu().numpy()
                print(f"    - Lớp: {class_name} (ID: {class_id}), Confidence: {confidence:.2f}, Tọa độ: {xyxy}")
        else:
            print("  Không tìm thấy đối tượng nào.")

def export_model(model_weights_path, export_format='onnx'):
    """Xuất model sang một định dạng khác (đã được cập nhật cho YOLOv11)."""
    print(f"--- Bắt đầu Xuất Model: {model_weights_path} sang định dạng {export_format} ---")
    if not os.path.exists(model_weights_path):
        print(f"Lỗi: File trọng số '{model_weights_path}' không tồn tại.")
        return

    model = YOLO(model_weights_path) # Sử dụng lớp YOLO gốc
    try:
        exported_model_path = model.export(
            format=export_format,
            imgsz=512,
            # half=True, # Cân nhắc, tùy thuộc vào YOLOv11 và định dạng export
            # dynamic=True, # Cân nhắc
        )
        print(f"--- Xuất Model Hoàn tất ---")
        print(f"Model đã xuất được lưu tại: {exported_model_path}")
    except Exception as e:
        print(f"Lỗi khi xuất model: {e}")

if __name__ == '__main__':
    TRAINED_MODEL_PATH = None

    # 1. HUẤN LUYỆN MODEL MỚI
    # Giả định bạn có file yolov11n.pt hoặc yolov11n.yaml tương ứng
    # và nó có thể được tải bởi `YOLO(BASE_MODEL_NAME)`
    # Nếu BASE_MODEL_NAME trỏ đến một file .yaml (ví dụ: 'yolov11n.yaml'),
    # model sẽ được xây dựng từ cấu hình đó.
    # Nếu trỏ đến .pt, nó sẽ tải trọng số pre-trained nếu có.
    print(f"Kiểm tra sự tồn tại của model cơ sở: {BASE_MODEL_NAME}")
    # Thêm một kiểm tra nhỏ nếu BASE_MODEL_NAME là file .pt và không tồn tại
    # Trong trường hợp đó, YOLO() sẽ tự động cố gắng tải xuống từ internet nếu đó là model chuẩn.
    # Nếu đó là file custom .pt, nó phải tồn tại cục bộ.
    
    best_pt_path = train_model()
    if best_pt_path and os.path.exists(best_pt_path):
        TRAINED_MODEL_PATH = best_pt_path
    else:
        # Fallback nếu train_model không trả về đường dẫn hoặc file không tồn tại
        # Cần một cách chắc chắn hơn để lấy đường dẫn, ví dụ, tạo thủ công từ PROJECT_NAME và EXPERIMENT_NAME_TRAIN
        # Dựa trên cấu trúc thư mục của Ultralytics: runs/detect/PROJECT_NAME/EXPERIMENT_NAME_TRAIN/weights/best.pt
        # Tuy nhiên, EXPERIMENT_NAME_TRAIN có thể được tự động thêm hậu tố (vd: _02, _03) nếu đã tồn tại.
        # Cách an toàn nhất là lấy từ `results.save_dir` sau khi train xong, như đã làm.
        # Nếu không chạy train, bạn phải đặt thủ công:
        print(f"Không thể xác định TRAINED_MODEL_PATH tự động sau khi train. Kiểm tra output của hàm train_model().")
        print(f"Giả sử đường dẫn mặc định nếu tên experiment không bị thay đổi:")
        assumed_path = os.path.join('runs', 'detect', PROJECT_NAME, EXPERIMENT_NAME_TRAIN, 'weights', 'best.pt')
        if os.path.exists(assumed_path):
            TRAINED_MODEL_PATH = assumed_path
            print(f"Sử dụng đường dẫn giả định: {TRAINED_MODEL_PATH}")
        else: # Ví dụ, nếu tên experiment đã có và Ultralytics tự động thêm số (vd: train_yolov11_012)
            # Đây là lúc bạn cần tìm thư mục experiment mới nhất.
            # Mã bên dưới chỉ là một ví dụ đơn giản, cần tinh chỉnh.
            try:
                project_runs_dir = os.path.join('runs', 'detect', PROJECT_NAME)
                if os.path.exists(project_runs_dir):
                    all_experiments = [d for d in os.listdir(project_runs_dir) if os.path.isdir(os.path.join(project_runs_dir, d)) and d.startswith(EXPERIMENT_NAME_TRAIN)]
                    if all_experiments:
                        latest_experiment = sorted(all_experiments)[-1] # Sắp xếp theo tên, không phải thời gian
                        potential_path = os.path.join(project_runs_dir, latest_experiment, 'weights', 'best.pt')
                        if os.path.exists(potential_path):
                            TRAINED_MODEL_PATH = potential_path
                            print(f"Tìm thấy model đã huấn luyện mới nhất tại: {TRAINED_MODEL_PATH}")
            except Exception as e:
                print(f"Lỗi khi cố gắng tìm model mới nhất: {e}")
        
        if not TRAINED_MODEL_PATH:
            print(f"Không tìm thấy model huấn luyện. Đặt TRAINED_MODEL_PATH thủ công nếu cần.")
            # TRAINED_MODEL_PATH = 'runs/detect/CADOT_YOLOv11/train_yolov11_01/weights/best.pt' # Cần đặt thủ công


    # 3. ĐÁNH GIÁ MODEL SAU KHI HUẤN LUYỆN
    if TRAINED_MODEL_PATH and os.path.exists(TRAINED_MODEL_PATH):
         validate_model(TRAINED_MODEL_PATH)
    else:
         print(f"Không tìm thấy model đã huấn luyện tại '{TRAINED_MODEL_PATH}' để đánh giá. Hãy huấn luyện trước.")


    # # 4. DỰ ĐOÁN TRÊN ẢNH/VIDEO/THƯ MỤC
    # if TRAINED_MODEL_PATH and os.path.exists(TRAINED_MODEL_PATH):
    #     image_to_predict = '/datausers3/kttv/tien/ClassificationProjectHimawari/test/CADOT_Dataset/valid/images/example_image_01.jpg'

    #     if not os.path.exists(image_to_predict) and image_to_predict.endswith(('.jpg', '.png', '.jpeg')):
    #          print(f"Cảnh báo: File ảnh '{image_to_predict}' không tồn tại. Tạo ảnh giả để demo.")
    #          try:
    #              from PIL import Image, ImageDraw
    #              # Đảm bảo thư mục tồn tại
    #              os.makedirs(os.path.dirname(image_to_predict), exist_ok=True)
    #              img = Image.new('RGB', (640, 480), color = 'red')
    #              d = ImageDraw.Draw(img)
    #              d.text((10,10), "Sample Image for YOLOv11", fill=(255,255,0))
    #              img.save(image_to_predict)
    #              print(f"Đã tạo ảnh giả tại: {image_to_predict}")
    #          except ImportError:
    #              print("Cần Pillow để tạo ảnh giả. Vui lòng cài đặt: pip install Pillow")
    #          except Exception as e:
    #              print(f"Không thể tạo ảnh giả: {e}")

    #     if os.path.exists(image_to_predict) or image_to_predict.startswith('http'):
    #         predict_with_model(TRAINED_MODEL_PATH, image_to_predict)
    #     else:
    #         print(f"Nguồn ảnh/video '{image_to_predict}' không hợp lệ hoặc không tồn tại cho việc dự đoán.")
    # else:
    #     print(f"Không tìm thấy model đã huấn luyện tại '{TRAINED_MODEL_PATH}' để dự đoán. Hãy huấn luyện trước.")


    # 5. XUẤT MODEL SANG ĐỊNH DẠNG KHÁC (ví dụ ONNX)
    # if TRAINED_MODEL_PATH and os.path.exists(TRAINED_MODEL_PATH):
    #     export_model(TRAINED_MODEL_PATH, export_format='onnx')
    # else:
    #     print(f"Không tìm thấy model đã huấn luyện tại '{TRAINED_MODEL_PATH}' để xuất. Hãy huấn luyện trước.")

    print("--- Hoàn thành tất cả các tác vụ được chọn ---")