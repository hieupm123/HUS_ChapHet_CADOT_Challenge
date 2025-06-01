import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import numpy as np

# --- CẤU HÌNH ---
GT_ANNOTATION_FILE = '../../data/CADOT_Dataset/valid/_annotations.coco.json'
BASE_INFERENCE_FOLDER = '../../data/evaluation_results'
OUTPUT_BEST_AP50_CSV_FILE = '../../results/metrics.csv'
OUTPUT_BEST_MODEL_PER_CLASS_CSV_FILE = '../../data/metrics.csv' # File CSV mới
# OUTPUT_ALL_METRICS_CSV_FILE = 'all_models_metrics_summary_ap50_no_small_object.csv'

# Danh sách TÊN lớp gốc, bao gồm cả lớp sẽ bị loại bỏ
ALL_POSSIBLE_CLASS_NAMES = [
    'basketball field', 'building', 'crosswalk', 'football field', 'graveyard',
    'large vehicle', 'medium vehicle', 'playground', 'roundabout', 'ship',
    'small vehicle', 'swimming pool', 'tennis court', 'train', 'small-object'
]

CLASS_TO_EXCLUDE = 'small-object' # Lớp cần loại bỏ

MODEL_FOLDERS = [
    'infer_yolov11_origin', 'infer_yolov11_v3', 'infer_yolov11_v5',
    'infer_yolov12_v3', 'infer_yolov12_v5'
]

# --- KẾT THÚC CẤU HÌNH ---

def get_coco_ap50(coco_eval_obj, gt_category_id_to_evaluate):
    """Trích xuất AP@IoU=0.50 cho một category_id cụ thể từ COCOeval."""
    precisions = coco_eval_obj.eval['precision']
    try:
        iou_thresholds = coco_eval_obj.params.iouThrs
        iou_50_idx = np.where(np.isclose(iou_thresholds, 0.50))[0]
        if not iou_50_idx.size:
            print("CẢNH BÁO: IoU threshold 0.50 không tìm thấy. Trả về 0.")
            return 0.0
        iou_50_idx = iou_50_idx[0]
    except Exception as e:
        print(f"Lỗi khi tìm index cho IoU 0.50: {e}. Trả về 0.")
        return 0.0

    try:
        # gt_category_id_to_evaluate phải có trong coco_eval.params.catIds
        # mà đã được thiết lập trước khi gọi evaluate()
        if gt_category_id_to_evaluate not in coco_eval_obj.params.catIds:
            # print(f"CẢNH BÁO: GT Category ID {gt_category_id_to_evaluate} không có trong catIds của COCOeval.params. Trả về 0.")
            return 0.0
        cat_idx_in_eval_params = coco_eval_obj.params.catIds.index(gt_category_id_to_evaluate)
    except ValueError:
        # Điều này không nên xảy ra nếu catIds được đặt chính xác và gt_category_id_to_evaluate là một trong số đó
        # print(f"CẢNH BÁO: GT Category ID {gt_category_id_to_evaluate} không tìm thấy trong coco_eval.params.catIds sau khi kiểm tra. Trả về 0.")
        return 0.0

    aind = coco_eval_obj.params.areaRngLbl.index('all')
    try:
        m_100_idx = coco_eval_obj.params.maxDets.index(100)
    except ValueError:
        if coco_eval_obj.params.maxDets:
            m_100_idx = len(coco_eval_obj.params.maxDets) - 1 # Sử dụng maxDets cuối cùng nếu 100 không có
        else: # Trường hợp params.maxDets trống
            print(f"CẢNH BÁO: COCOeval.params.maxDets trống. Trả về 0 cho cat_id {gt_category_id_to_evaluate}")
            return 0.0

    # P = precisions[T, R, K, A, M]
    # T: iouThrs index (iou_50_idx)
    # R: recallThrs index (dùng tất cả: :)
    # K: catIds index (cat_idx_in_eval_params)
    # A: areaRngLbl index (aind)
    # M: maxDets index (m_100_idx)
    s = precisions[iou_50_idx, :, cat_idx_in_eval_params, aind, m_100_idx]
    ap50 = np.mean(s[s > -1]) if s[s > -1].size > 0 else 0.0
    return ap50

def main():
    # Tạo danh sách các lớp thực sự cần xử lý
    class_names_to_process = [name for name in ALL_POSSIBLE_CLASS_NAMES if name != CLASS_TO_EXCLUDE]
    if not class_names_to_process:
        print(f"LỖI: Sau khi loại bỏ '{CLASS_TO_EXCLUDE}', không còn lớp nào để xử lý. Dừng chương trình.")
        return
    print(f"Các lớp sẽ được xử lý (sau khi loại bỏ '{CLASS_TO_EXCLUDE}'): {class_names_to_process}")


    print(f"Đang tải ground truth annotations từ: {GT_ANNOTATION_FILE}")
    if not os.path.exists(GT_ANNOTATION_FILE):
        print(f"LỖI: Không tìm thấy file ground truth '{GT_ANNOTATION_FILE}'. Dừng chương trình.")
        return
    try:
        coco_gt = COCO(GT_ANNOTATION_FILE)
    except Exception as e:
        print(f"LỖI khi tải file ground truth: {e}. Dừng chương trình.")
        return

    detection_class_name_to_gt_id = {}
    gt_name_to_gt_id_from_coco_file = {cat['name']: cat['id'] for cat in coco_gt.dataset.get('categories', [])}

    print("Ánh xạ tên lớp (đã lọc) sang GT Category ID:")
    for class_name_in_list in class_names_to_process:
        if class_name_in_list in gt_name_to_gt_id_from_coco_file:
            gt_id = gt_name_to_gt_id_from_coco_file[class_name_in_list]
            detection_class_name_to_gt_id[class_name_in_list] = gt_id
            print(f"  '{class_name_in_list}' -> GT ID: {gt_id}")
        else:
            print(f"  CẢNH BÁO: Tên lớp '{class_name_in_list}' (đã lọc) KHÔNG TÌM THẤY trong '{GT_ANNOTATION_FILE}'. Sẽ bỏ qua lớp này khỏi việc map.")

    if not detection_class_name_to_gt_id:
        print("LỖI: Không có lớp nào (đã lọc) khớp với categories trong GT. Dừng chương trình.")
        return

    active_class_names = list(detection_class_name_to_gt_id.keys())
    gt_ids_to_actually_evaluate = sorted(list(detection_class_name_to_gt_id.values()))

    if not gt_ids_to_actually_evaluate:
        print("LỖI: Không có GT IDs nào để đánh giá sau khi lọc và map. Dừng chương trình.")
        return

    all_models_ap50_data = {}

    for model_folder_name in MODEL_FOLDERS:
        model_folder_path = os.path.join(BASE_INFERENCE_FOLDER, model_folder_name)
        print(f"\nĐang xử lý model: {model_folder_name} trong thư mục: {model_folder_path}")

        current_model_class_ap50_scores = {name: 0.0 for name in active_class_names} # Khởi tạo với 0

        if not os.path.isdir(model_folder_path):
            print(f"  CẢNH BÁO: Thư mục model '{model_folder_path}' không tồn tại. AP50 cho tất cả các lớp của model này sẽ là 0.")
            all_models_ap50_data[model_folder_name] = current_model_class_ap50_scores
            continue

        model_all_predictions_for_eval = []
        for class_name_for_file, gt_category_id in detection_class_name_to_gt_id.items():
            detection_file_leafname = f"detections_class_{class_name_for_file.replace(' ', '_')}.json"
            detection_file_path = os.path.join(model_folder_path, detection_file_leafname)

            if os.path.exists(detection_file_path):
                try:
                    with open(detection_file_path, 'r') as f:
                        class_specific_raw_predictions = json.load(f)
                    for pred in class_specific_raw_predictions:
                        if not all(k in pred for k in ['image_id', 'bbox', 'score']):
                            # print(f"    Cảnh báo: Dự đoán không hợp lệ trong {detection_file_path}: {pred}. Bỏ qua.")
                            continue
                        processed_pred = {
                            "image_id": pred['image_id'], "category_id": gt_category_id,
                            "bbox": pred['bbox'], "score": pred['score']
                        }
                        model_all_predictions_for_eval.append(processed_pred)
                except Exception as e:
                    print(f"    Lỗi khi đọc file '{detection_file_path}': {e}. Bỏ qua class này cho model này.")
            # else:
                # print(f"    Thông báo: File detection '{detection_file_path}' không tồn tại cho model {model_folder_name}.")


        if not model_all_predictions_for_eval:
            print(f"  Không có prediction hợp lệ nào được tải cho model '{model_folder_name}'. AP50 cho tất cả các lớp của model này sẽ là 0.")
            all_models_ap50_data[model_folder_name] = current_model_class_ap50_scores
            continue

        valid_image_ids = set(coco_gt.getImgIds())
        # Lọc dự đoán chỉ cho các image_id có trong tập GT
        filtered_predictions = [p for p in model_all_predictions_for_eval if p['image_id'] in valid_image_ids]

        if not filtered_predictions:
            print(f"  Không có prediction nào thuộc các image_id hợp lệ cho model '{model_folder_name}' sau khi lọc. AP50 cho tất cả các lớp của model này sẽ là 0.")
            all_models_ap50_data[model_folder_name] = current_model_class_ap50_scores
            continue
            
        try:
            coco_dt = coco_gt.loadRes(filtered_predictions)
        except Exception as e:
            print(f"  Lỗi khi loadRes cho model '{model_folder_name}': {e}. AP50 cho tất cả các lớp của model này sẽ là 0.")
            all_models_ap50_data[model_folder_name] = current_model_class_ap50_scores
            continue

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = gt_ids_to_actually_evaluate # Chỉ đánh giá trên các lớp đã chọn

        print(f"  Đang chạy COCO evaluation cho model '{model_folder_name}' trên {len(gt_ids_to_actually_evaluate)} lớp...")
        coco_eval.evaluate()
        coco_eval.accumulate()

        for class_name_key, gt_id_value in detection_class_name_to_gt_id.items():
            if gt_id_value in gt_ids_to_actually_evaluate: # Đảm bảo gt_id này nằm trong danh sách đánh giá
                ap50_score_for_class = get_coco_ap50(coco_eval, gt_id_value)
                current_model_class_ap50_scores[class_name_key] = round(ap50_score_for_class, 4)
            # else: class_name_key đã được khởi tạo với 0.0
        
        all_models_ap50_data[model_folder_name] = current_model_class_ap50_scores
        print(f"  AP50 scores for {model_folder_name} (for processed classes): {current_model_class_ap50_scores}")

    # ---- Tìm AP50 lớn nhất và model tương ứng cho mỗi lớp ----
    best_ap50_per_class_with_model_info = []
    best_ap50_scores_for_hybrid_map50 = []

    for class_name in active_class_names:
        max_ap50_for_this_class = -1.0
        best_model_for_this_class = "N/A"

        for model_folder in MODEL_FOLDERS:
            if model_folder in all_models_ap50_data and \
               class_name in all_models_ap50_data[model_folder]:
                current_ap50 = all_models_ap50_data[model_folder][class_name]
                if current_ap50 > max_ap50_for_this_class:
                    max_ap50_for_this_class = current_ap50
                    best_model_for_this_class = model_folder
        
        # Nếu max_ap50_for_this_class vẫn là -1 (không có model nào có score cho lớp này),
        # actual_best_ap50 sẽ là 0.0 và best_model_for_this_class sẽ là "N/A".
        actual_best_ap50 = round(max_ap50_for_this_class, 4) if max_ap50_for_this_class > -1.0 else 0.0
        
        # Nếu actual_best_ap50 là 0.0 vì max_ap50_for_this_class là -1.0 (không có dữ liệu), model vẫn là N/A
        # Nếu actual_best_ap50 là 0.0 vì max_ap50_for_this_class thực sự là 0.0, best_model_for_this_class đã được set.
        if max_ap50_for_this_class <= -1.0 : # Chính xác hơn, nếu không có model nào cho điểm
            best_model_for_this_class = "N/A"

        best_ap50_per_class_with_model_info.append({
            "Class_Name": class_name,
            "Best_AP50_Score": actual_best_ap50,
            "Best_Model_Folder": best_model_for_this_class
        })
        best_ap50_scores_for_hybrid_map50.append(actual_best_ap50)

    # --- Lưu file CSV cho best model per class ---
    if best_ap50_per_class_with_model_info:
        best_model_df = pd.DataFrame(best_ap50_per_class_with_model_info)
        try:
            output_dir = os.path.dirname(OUTPUT_BEST_MODEL_PER_CLASS_CSV_FILE)
            if output_dir: # Đảm bảo thư mục tồn tại nếu có
                os.makedirs(output_dir, exist_ok=True)
            
            columns_for_best_model_export = ["Class_Name", "Best_AP50_Score", "Best_Model_Folder"]
            best_model_df.to_csv(OUTPUT_BEST_MODEL_PER_CLASS_CSV_FILE, columns=columns_for_best_model_export, index=False)
            print(f"\nBảng model tốt nhất cho mỗi lớp (không có '{CLASS_TO_EXCLUDE}') đã được lưu vào: {OUTPUT_BEST_MODEL_PER_CLASS_CSV_FILE}")
            print(f"\nKết quả Best Model per Class (không có '{CLASS_TO_EXCLUDE}'):")
            print(best_model_df[columns_for_best_model_export].to_string())
        except Exception as e:
            print(f"LỖI khi lưu file CSV '{OUTPUT_BEST_MODEL_PER_CLASS_CSV_FILE}': {e}")
    else:
        print(f"\nKhông có kết quả Best Model per Class nào (không có '{CLASS_TO_EXCLUDE}') để lưu.")

    # --- Tính mAP50 "Hybrid" và lưu file metrics.csv gốc ---
    hybrid_mAP50 = 0.0
    if best_ap50_scores_for_hybrid_map50: # Kiểm tra nếu danh sách không rỗng
        hybrid_mAP50 = round(np.mean(best_ap50_scores_for_hybrid_map50), 4)
        print(f"\nHybrid mAP50 (mean of best AP50s for processed classes, excluding '{CLASS_TO_EXCLUDE}'): {hybrid_mAP50}")
    else:
        print(f"\nKhông thể tính Hybrid mAP50 (excluding '{CLASS_TO_EXCLUDE}') vì không có AP50 nào được ghi nhận cho các lớp đã xử lý.")

    # Tạo DataFrame cho file metrics.csv (OUTPUT_BEST_AP50_CSV_FILE)
    # Chỉ chứa "Class_Name" và "Best_AP50_Score", và dòng mAP50 hybrid
    records_for_original_metrics_csv = []
    for record in best_ap50_per_class_with_model_info: # Sử dụng lại dữ liệu đã tính
        records_for_original_metrics_csv.append({
            "Class_Name": record["Class_Name"],
            "Best_AP50_Score": record["Best_AP50_Score"]
        })
    
    if records_for_original_metrics_csv: # Nếu có lớp nào được xử lý
        best_ap50_df_original = pd.DataFrame(records_for_original_metrics_csv)
        
        if best_ap50_scores_for_hybrid_map50: # Chỉ thêm dòng mAP50 nếu có giá trị để tính
            hybrid_map50_row = pd.DataFrame([{
                "Class_Name": f"OVERALL (Hybrid mAP50, excl. '{CLASS_TO_EXCLUDE}')",
                "Best_AP50_Score": hybrid_mAP50
            }])
            best_ap50_df_original = pd.concat([best_ap50_df_original, hybrid_map50_row], ignore_index=True)
            
        try:
            output_dir = os.path.dirname(OUTPUT_BEST_AP50_CSV_FILE)
            if output_dir:
                 os.makedirs(output_dir, exist_ok=True)

            columns_to_export_original = ["Class_Name", "Best_AP50_Score"]
            best_ap50_df_original.to_csv(OUTPUT_BEST_AP50_CSV_FILE, columns=columns_to_export_original, index=False)
            print(f"\nBảng AP50 lớn nhất cho mỗi lớp (không có '{CLASS_TO_EXCLUDE}', bao gồm Hybrid mAP50) đã được lưu vào: {OUTPUT_BEST_AP50_CSV_FILE}")
            print(f"\nKết quả Best AP50 per Class (không có '{CLASS_TO_EXCLUDE}', bao gồm Hybrid mAP50):")
            print(best_ap50_df_original[columns_to_export_original].to_string())
        except Exception as e:
            print(f"LỖI khi lưu file CSV '{OUTPUT_BEST_AP50_CSV_FILE}': {e}")
    else:
        print(f"\nKhông có kết quả AP50 nào (không có '{CLASS_TO_EXCLUDE}') để lưu vào '{OUTPUT_BEST_AP50_CSV_FILE}'.")


if __name__ == '__main__':
    critical_error = False
    if not os.path.exists(GT_ANNOTATION_FILE):
        print(f"Lỗi nghiêm trọng: File ground truth '{GT_ANNOTATION_FILE}' không tồn tại.")
        critical_error = True
    if not ALL_POSSIBLE_CLASS_NAMES:
        print("Lỗi nghiêm trọng: Danh sách 'ALL_POSSIBLE_CLASS_NAMES' đang trống.")
        critical_error = True
    if not MODEL_FOLDERS:
        print("Lỗi nghiêm trọng: Danh sách 'MODEL_FOLDERS' đang trống.")
        critical_error = True
    
    if not critical_error:
        main()
    else:
        print("Do có lỗi nghiêm trọng, chương trình sẽ không chạy.")