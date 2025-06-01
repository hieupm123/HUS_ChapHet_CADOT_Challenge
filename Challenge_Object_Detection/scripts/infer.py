import os # Import module os

os.chdir("../model/infer/")


# Danh sách các file cần chạy tuần tự
scripts = [
    "get_infer_test_by_class.py",
    "mergeJson.py",
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