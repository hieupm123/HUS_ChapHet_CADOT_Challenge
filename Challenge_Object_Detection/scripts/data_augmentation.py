import subprocess

# Danh sách các file cần chạy tuần tự
scripts = [
    "../model/argumentdata_split_class.py",
    "../model/argumentdata_routine.py",
    "../model/argumentdata_change_color.py",
    "../model/argumentdata_2_3.py",
    "../model/argumentdata_new_background.py",
    "../model/fixannot_v1.py",
    "../model/get_new_data_v1.py",
    "../model/get_new_data_v2.py",
    "../model/get_new_data_v3.py",
    "../model/get_new_data_v4.py",
    "../model/get_new_data_v5.py",
    "../model/merge_new_data_v1.py",
    "../model/fixannot_v2.py",
    "../model/get_new_data_v6.py",
    "../model/merge_new_data_v2.py",
    "../model/cocoToYolo.py"
]

for script in scripts:
    print(f"Đang chạy {script} ...")
    result = subprocess.run(["python3", script])
    if result.returncode != 0:
        print(f"Script {script} gặp lỗi. Dừng quá trình.")
        break
    else:
        print(f"Hoàn thành {script}\n")
