import os # Import module os

os.chdir("../model/train/")

folders_to_delete = [
    "CADOT_YOLOv11_Origin",
    "CADOT_YOLOv11_v3",
    "CADOT_YOLOv11_v5",
    "CADOT_YOLOv12_v3",
    "CADOT_YOLOv12_v5"
]

for folder in folders_to_delete:
    if os.path.exists(folder) and os.path.isdir(folder):
        os.system(f'rm -rf "{folder}"')


# Danh sách các file cần chạy tuần tự
scripts = [
    "yolov11_origin.py",
    "yolov11_v3.py",
    "yolov11_v5.py",
    "yolov12_v3.py",
    "yolov12_v5.py",
]


for script in scripts:

    print(f"Đang chạy {script} ...")
    command = f"python3 {script}"
    print(f"Thực thi lệnh: {command}") # Tùy chọn: in lệnh để debug
    return_code = os.system(command)

    if return_code != 0:
        print(f"Script {script} gặp lỗi (mã lỗi: {return_code}). Dừng quá trình.")
        break
    else:
        print(f"Hoàn thành {script}\n")

print("Hoàn tất tất cả các script (hoặc dừng do lỗi).")