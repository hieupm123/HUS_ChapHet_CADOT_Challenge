import os
from PIL import Image
# Nếu chưa cài đặt Pillow, hãy chạy: pip install Pillow

def rotate_images_in_folder(input_folder, output_folder, rotations_to_create):
    """
    Duyệt qua tất cả ảnh trong input_folder, tạo các phiên bản xoay
    và lưu chúng vào output_folder.

    Args:
        input_folder (str): Đường dẫn đến thư mục chứa ảnh gốc.
        output_folder (str): Thư mục để lưu tất cả các ảnh đã xoay.
        rotations_to_create (dict): Dictionary chứa tên hậu tố và góc xoay (độ).
                                    Ví dụ: {"rot90": 90, "rot-90": -90, "rot180": 180}
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    print(f"Thư mục lưu ảnh xoay: {output_folder}")

    # Các định dạng ảnh phổ biến cần xử lý
    allowed_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    # Đếm số ảnh đã xử lý
    processed_count = 0
    error_count = 0

    list_of_files = os.listdir(input_folder)

    print(f"Bắt đầu quét thư mục: {input_folder}")

    # Duyệt qua từng file trong thư mục
    for filename in list_of_files:
        # Chỉ xử lý các file có phần mở rộng là ảnh
        if filename.lower().endswith(allowed_extensions):
            input_image_path = os.path.join(input_folder, filename)

            # Đảm bảo rằng đó là file chứ không phải thư mục con
            if os.path.isfile(input_image_path):
                print(f"\nĐang xử lý ảnh: {filename}")
                # Mở ảnh gốc
                img = Image.open(input_image_path)
                base_name = os.path.splitext(filename)[0]
                extension = os.path.splitext(filename)[1]
                # Tạo các phiên bản xoay cho ảnh hiện tại
                for name_suffix, angle in rotations_to_create.items():
                    print(f"  - Đang xoay {angle} độ...")
                    try:
                        # Dùng cho Pillow phiên bản mới
                        rotated_img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
                    except AttributeError:
                        # Fallback cho Pillow phiên bản cũ hơn
                        rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
                    # Tạo tên file output
                    output_filename = f"{base_name}_{name_suffix}{extension}"
                    output_path = os.path.join(output_folder, output_filename)
                    # Lưu ảnh đã xoay
                    rotated_img.save(output_path)
                    print(f"    Đã lưu: {output_filename}")
                processed_count += 1


    print("\n--------------------")
    print("Hoàn tất xử lý thư mục.")
    print(f"Tổng số ảnh gốc đã xử lý thành công: {processed_count}")
    print(f"Số ảnh gặp lỗi khi xử lý: {error_count}")
    print(f"Các ảnh xoay đã được lưu vào: {output_folder}")
    print("--------------------")

# Các góc xoay cần tạo (và hậu tố tên file tương ứng)
ROTATIONS = {
    "origin": 0,
    "rot90": 90,
    "rotN90": -90, # Sử dụng rotN90 thay vì rot-90 để tên file thân thiện hơn
    "rot180": 180
}

# --- Chạy script ---
if __name__ == "__main__":
    rotate_images_in_folder("../data/split_folder_all_classes/basketball_field", "../data/outargument/basketball_field", ROTATIONS)
    rotate_images_in_folder("../data/split_folder_all_classes/football_field", "../data/outargument/football_field", ROTATIONS)
    rotate_images_in_folder("../data/split_folder_all_classes/graveyard", "../data/outargument/graveyard", ROTATIONS)
    rotate_images_in_folder("../data/split_folder_all_classes/playground", "../data/outargument/playground", ROTATIONS)
    rotate_images_in_folder('../data/split_folder_all_classes/roundabout', "../data/outargument/roundabout", ROTATIONS)
    # rotate_images_in_folder('../data/split_folder_all_classes/crosswalk', "../data/outargument/crosswalk", ROTATIONS)
    rotate_images_in_folder('../data/split_folder_all_classes/ship', "../data/outargument/ship", ROTATIONS)
    rotate_images_in_folder('../data/split_folder_all_classes/swimming_pool', "../data/outargument/swimming_pool", ROTATIONS)
    rotate_images_in_folder('../data/split_folder_all_classes/tennis_court', "../data/outargument/tennis_court", ROTATIONS)
    rotate_images_in_folder('../data/split_folder_all_classes/train', "../data/outargument/train", ROTATIONS)