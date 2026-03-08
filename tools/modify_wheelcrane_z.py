import os
import json
from tqdm import tqdm

# label_root = "/home/baojiali/Downloads/disk1/data/lightwheel_data/label"
label_root = "/home/baojiali/Downloads/disk2/data/qingdao_8lidar_data/label"
target_subtypes = ["WheelCrane","RailCrane"]

# 初始化计数器
label_file_count = 0
subtype_counts = {subtype: 0 for subtype in target_subtypes}
modified_obj_count = 0
modified_file_count = 0

# 先收集所有 json 文件路径
json_files = []
for root, _, files in os.walk(label_root):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file))

# 遍历时加进度条
for json_path in tqdm(json_files, desc="Processing JSON files"):
    label_file_count += 1
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        modified = False
        for obj in data:
            subtype = obj.get("subtype")
            if subtype in subtype_counts:
                subtype_counts[subtype] += 1
                if "lwh" in obj and len(obj["lwh"]) == 3 and "xyz" in obj and len(obj["xyz"]) == 3:
                    old_h = obj["lwh"][2]
                    bottom_z = obj["xyz"][2] - old_h / 2.0
                    new_h = 4.0
                    new_center_z = bottom_z + new_h / 2.0

                    obj["lwh"][2] = new_h
                    obj["xyz"][2] = new_center_z

                    modified = True
                    modified_obj_count += 1

        if modified:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            modified_file_count += 1

    except Exception as e:
        print(f"Failed to read {json_path}: {e}")

# ✅ 输出结果
print(f"\nTotal label files: {label_file_count}")
print(f"Modified files: {modified_file_count}")
print(f"Modified objects: {modified_obj_count}")
print("Subtype counts:")
for subtype, count in subtype_counts.items():
    print(f"  {subtype}: {count}")
