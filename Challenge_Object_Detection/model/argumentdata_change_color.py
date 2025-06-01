from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os
import time

# --- Hàm tạo biến thể màu sắc (sửa đổi để nhận đối tượng Image) ---

def change_saturation(img_object_rgb, output_path, factor):
    """
    Thay đổi Độ bão hòa (Saturation) của đối tượng ảnh PIL đã cho.
    img_object_rgb: Đối tượng PIL Image đã convert sang 'RGB'.
    output_path: Đường dẫn lưu ảnh kết quả.
    factor: Hệ số thay đổi. < 1.0: giảm, > 1.0: tăng.
    """
    try:
        enhancer = ImageEnhance.Color(img_object_rgb)
        img_saturated = enhancer.enhance(factor)
        img_saturated.save(output_path)
        # print(f"  Đã lưu Saturation (factor {factor:.1f}): {output_path}") # Ít chi tiết hơn để tránh quá nhiều output
    except Exception as e:
        print(f"Lỗi khi thay đổi Saturation cho {output_path}: {e}")

def change_brightness(img_object_rgb, output_path, factor):
    """
    Thay đổi Độ sáng (Brightness) của đối tượng ảnh PIL đã cho.
    img_object_rgb: Đối tượng PIL Image đã convert sang 'RGB'.
    output_path: Đường dẫn lưu ảnh kết quả.
    factor: Hệ số thay đổi. < 1.0: tối hơn, > 1.0: sáng hơn.
    """
    try:
        enhancer = ImageEnhance.Brightness(img_object_rgb)
        img_bright = enhancer.enhance(factor)
        img_bright.save(output_path)
        # print(f"  Đã lưu Brightness (factor {factor:.1f}): {output_path}")
    except Exception as e:
        print(f"Lỗi khi thay đổi Brightness cho {output_path}: {e}")

def apply_sepia(img_object_rgb, output_path):
    """
    Áp dụng bộ lọc màu Sepia cho đối tượng ảnh PIL đã cho.
    img_object_rgb: Đối tượng PIL Image đã convert sang 'RGB'.
    output_path: Đường dẫn lưu ảnh kết quả.
    """
    try:
        # Công thức tạo màu Sepia
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        img_array = np.array(img_object_rgb, dtype=np.float64)
        sepia_img_array = img_array.dot(sepia_filter.T)
        sepia_img_array = np.clip(sepia_img_array, 0, 255)
        img_sepia = Image.fromarray(sepia_img_array.astype(np.uint8))
        img_sepia.save(output_path)
        # print(f"  Đã lưu Sepia: {output_path}")
    except Exception as e:
        print(f"Lỗi khi áp dụng Sepia cho {output_path}: {e}")

def apply_grayscale(img_object, output_path):
    """
    Chuyển đối tượng ảnh PIL đã cho sang thang độ xám.
    img_object: Đối tượng PIL Image gốc (chưa cần convert).
    output_path: Đường dẫn lưu ảnh kết quả.
    """
    try:
        img_gray = img_object.convert('L')
        img_gray.save(output_path)
        # print(f"  Đã lưu Grayscale: {output_path}")
    except Exception as e:
        print(f"Lỗi khi chuyển sang Grayscale cho {output_path}: {e}")


# --- Hàm xử lý thư mục ---
def process_directory(input_dir, output_dir):
    """
    Duyệt qua input_dir, tìm ảnh, áp dụng biến đổi và lưu vào output_dir.
    """
    print(f"Bắt đầu xử lý thư mục: {input_dir}")
    print(f"Kết quả sẽ được lưu vào: {output_dir}")
    processed_count = 0
    skipped_count = 0
    start_time = time.time()

    # Các đuôi file ảnh hợp lệ (chữ thường)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # Duyệt qua tất cả các file và thư mục con trong input_dir
    for root, dirs, files in os.walk(input_dir):
        print(f"Đang quét thư mục: {root}")
        for filename in files:
            # Kiểm tra đuôi file (không phân biệt hoa thường)
            if filename.lower().endswith(valid_extensions):
                input_image_path = os.path.join(root, filename)

                try:
                    # Mở ảnh gốc một lần
                    with Image.open(input_image_path) as img_original:
                        print(f"  -> Đang xử lý: {filename}")
                        processed_count += 1

                        # Chuyển sang RGB để xử lý các biến thể màu
                        # (Lưu ý: Grayscale sẽ tự convert về L)
                        try:
                             img_rgb = img_original.convert('RGB')
                        except Exception as convert_err:
                             print(f"    Lỗi khi convert sang RGB cho {filename}, bỏ qua: {convert_err}")
                             skipped_count += 1
                             continue # Chuyển sang file tiếp theo nếu không convert được


                        # Xác định cấu trúc thư mục con trong output
                        relative_path = os.path.relpath(root, input_dir) # Lấy đường dẫn tương đối
                        current_output_dir = os.path.join(output_dir, relative_path)

                        # Tạo thư mục output nếu chưa có
                        os.makedirs(current_output_dir, exist_ok=True)

                        # Lấy tên file gốc và đuôi file
                        base_name, file_ext = os.path.splitext(filename)

                        # --- Tạo và lưu các biến thể ---

                        # 1. Sepia
                        output_path_sepia = os.path.join(current_output_dir, f"{base_name}_sepia{file_ext}")
                        apply_sepia(img_rgb, output_path_sepia)

                        # 2. Saturation High
                        output_path_sat_high = os.path.join(current_output_dir, f"{base_name}_saturation_high{file_ext}")
                        change_saturation(img_rgb, output_path_sat_high, 1.8) # Factor = 1.8 (có thể điều chỉnh)

                        # 3. Saturation Low
                        output_path_sat_low = os.path.join(current_output_dir, f"{base_name}_saturation_low{file_ext}")
                        change_saturation(img_rgb, output_path_sat_low, 0.3) # Factor = 0.3 (có thể điều chỉnh)

                        # 4. Grayscale (sử dụng img_original để convert('L'))
                        output_path_gray = os.path.join(current_output_dir, f"{base_name}_grayscale{file_ext}")
                        apply_grayscale(img_original, output_path_gray)

                        # 5. Brightness Low
                        output_path_bright_low = os.path.join(current_output_dir, f"{base_name}_brightness_low{file_ext}")
                        change_brightness(img_rgb, output_path_bright_low, 0.6) # Factor = 0.6 (có thể điều chỉnh)

                        # 6. Brightness High
                        output_path_bright_high = os.path.join(current_output_dir, f"{base_name}_brightness_high{file_ext}")
                        change_brightness(img_rgb, output_path_bright_high, 1.4) # Factor = 1.4 (có thể điều chỉnh)

                except (IOError, SyntaxError) as e:
                    # Bỏ qua nếu file ảnh bị lỗi hoặc không đọc được
                    print(f"  Lỗi: Không thể mở hoặc xử lý file {input_image_path}. Bỏ qua. Lỗi: {e}")
                    skipped_count += 1
                except Exception as general_e:
                    # Bắt các lỗi không mong muốn khác
                    print(f"  Lỗi không xác định khi xử lý {input_image_path}. Bỏ qua. Lỗi: {general_e}")
                    skipped_count += 1

            # else: # Bỏ qua file không phải ảnh
            #     print(f"  Bỏ qua file không phải ảnh: {filename}")


    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- Hoàn Thành Xử Lý ---")
    print(f"Tổng số file ảnh đã xử lý: {processed_count}")
    print(f"Tổng số file bị bỏ qua (lỗi đọc/convert): {skipped_count}")
    print(f"Tổng thời gian thực thi: {total_time:.2f} giây")
    print(f"Kết quả được lưu trong thư mục: {output_dir}")

# --- Chương trình chính ---
if __name__ == "__main__":
    # --- Cấu hình ---
    input_base_dir = "../data/outargument" # <<<=== THƯ MỤC GỐC chứa ảnh
    output_base_dir = "../data/outargument_color_variants" # <<<=== THƯ MỤC ĐỂ LƯU KẾT QUẢ

    # Kiểm tra xem thư mục input có tồn tại không
    if not os.path.isdir(input_base_dir):
        print(f"Lỗi: Thư mục đầu vào '{input_base_dir}' không tồn tại hoặc không phải là thư mục.")
    else:
        # Tạo thư mục output cơ sở nếu chưa có
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"Đã đảm bảo thư mục output cơ sở tồn tại: {output_base_dir}")

        # Bắt đầu quá trình xử lý
        process_directory(input_base_dir, output_base_dir)