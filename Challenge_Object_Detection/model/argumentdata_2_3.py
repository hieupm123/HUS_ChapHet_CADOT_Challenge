import os
from PIL import Image
import glob
import shutil # Thêm thư viện shutil để copy file

# --- Cấu hình ---
INPUT_DIRS = [
    "../data/outargument_color_variants/basketball_field",
    "../data/outargument_color_variants/football_field",
    "../data/outargument_color_variants/graveyard",
    "../data/outargument_color_variants/playground",
    "../data/outargument_color_variants/roundabout",
    "../data/outargument_color_variants/swimming_pool",
    "../data/outargument_color_variants/tennis_court"
]
# Thư mục gốc để lưu các ảnh biến thể và ảnh gốc.
OUTPUT_BASE_DIR = "../data/out_data_with_new" # Đổi tên để rõ ràng hơn

# Các loại biến thể và cách tính toán vùng crop (left, upper, right, lower)
# (width, height) là kích thước ảnh gốc
VARIANT_CROPS = {
    "top_2_3": lambda w, h: (0, 0, w, h * 2 // 3),
    "bottom_2_3": lambda w, h: (0, h // 3, w, h),
    "left_2_3": lambda w, h: (0, 0, w * 2 // 3, h),
    "right_2_3": lambda w, h: (w // 3, 0, w, h),
}

# Tên thư mục con để lưu ảnh gốc
ORIGINAL_FOLDER_NAME = "original"

# Các định dạng ảnh được hỗ trợ (thêm nếu cần)
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff"]
# --- Kết thúc cấu hình ---

def create_image_variants_and_copy_originals(input_dir, output_base_dir):
    """
    Tạo các biến thể ảnh, copy ảnh gốc từ một thư mục đầu vào
    và lưu vào thư mục đầu ra.
    """
    category_name = os.path.basename(input_dir) # Tên thư mục con (vd: basketball_fields)
    print(f"Processing category: {category_name}")

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper()))) # Cho cả chữ hoa

    if not image_files:
        print(f"  No images found in {input_dir} with specified extensions.")
        return

    # Tạo thư mục để lưu ảnh gốc cho category này
    # Ví dụ: ./generated_cropped_variants_and_originals/basketball_fields/original/
    original_output_dir_for_category = os.path.join(output_base_dir, category_name)
    os.makedirs(original_output_dir_for_category, exist_ok=True)

    for img_path in image_files:
        try:
            original_filename = os.path.basename(img_path)
            filename_no_ext, ext = os.path.splitext(original_filename)

            # 1. Copy ảnh gốc
            original_dest_path = os.path.join(original_output_dir_for_category, original_filename)
            shutil.copy2(img_path, original_dest_path) # copy2 giữ lại metadata
            # print(f"  Copied original: {original_dest_path}")

            # 2. Tạo các biến thể crop
            img = Image.open(img_path)
            width, height = img.size

            for variant_name, crop_func in VARIANT_CROPS.items():
                # Tạo thư mục đầu ra cho từng loại biến thể
                # Ví dụ: ./generated_cropped_variants_and_originals/basketball_fields/top_2_3/
                variant_output_dir = os.path.join(output_base_dir, category_name)
                os.makedirs(variant_output_dir, exist_ok=True)

                # Tính toán vùng crop
                crop_box = crop_func(width, height)

                # Crop ảnh
                cropped_img = img.crop(crop_box)

                # Tên file đầu ra (giữ nguyên tên gốc, lưu trong thư mục con của biến thể)
                output_filename = original_filename
                output_path = os.path.join(variant_output_dir, output_filename)

                # Lưu ảnh đã crop
                if img.format and img.format.upper() == "JPEG":
                    cropped_img.save(output_path, quality=95)
                else:
                    # Pillow cố gắng suy ra định dạng từ phần mở rộng của output_path
                    # Nếu không có format rõ ràng, cần xử lý thêm hoặc chỉ định format khi save
                    try:
                        cropped_img.save(output_path)
                    except ValueError as ve:
                        print(f"  Warning: Could not determine format for {output_path}. Trying PNG. Error: {ve}")
                        try:
                            output_path_png = os.path.splitext(output_path)[0] + ".png"
                            cropped_img.save(output_path_png, "PNG")
                            # print(f"  Saved as PNG: {output_path_png}")
                        except Exception as save_err:
                            print(f"  Error saving {output_path} (or as PNG): {save_err}")

                # print(f"  Saved variant: {output_path}")
            img.close() # Đóng file ảnh sau khi xử lý xong các biến thể

        except FileNotFoundError:
            print(f"  Error: File not found {img_path}")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
    print(f"Finished processing category: {category_name}\n")


if __name__ == "__main__":
    # Tạo thư mục gốc đầu ra nếu chưa có
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"Output will be saved in: {os.path.abspath(OUTPUT_BASE_DIR)}\n")

    for input_folder in INPUT_DIRS:
        if os.path.isdir(input_folder):
            create_image_variants_and_copy_originals(input_folder, OUTPUT_BASE_DIR)
        else:
            print(f"Warning: Input directory not found: {input_folder}")

    print("All processing complete.")