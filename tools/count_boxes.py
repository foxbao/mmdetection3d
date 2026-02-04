import os
import json
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
# 你的15个目标类别（label + subtype组合）
TARGET_CLASSES = {
    'Car', 'ConstructionVehicle', 'ContainerForklift', 'Crane', 'Forklift',
    'IGV-Empty', 'IGV-Full', 'Lorry', 'Trailer-Empty', 'Trailer-Full',
    'Truck', 'WheelCrane', 'Pedestrian', 'Cone', 'OtherVehicle'
}
# ✅ 计算均值和标准差（已存在）
def compute_mean_std(data_list):
    if not data_list:
        return 0.0, 0.0
    n = len(data_list)
    mean = sum(data_list) / n
    var = sum((x - mean) ** 2 for x in data_list) / n
    std = math.sqrt(var)
    return mean, std

# ✅ 新增：用于合并所有尺寸数据
def merge_label_dims(label_dims_list):
    merged = defaultdict(list)
    for label_dims in label_dims_list:
        for k, dims_list in label_dims.items():
            merged[k].extend(dims_list)
    return merged

# ✅ 新增：统计尺寸（lwh）的均值和标准差
def compute_lwh_stats(dim_list):
    if not dim_list:
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
    l_list, w_list, h_list = zip(*dim_list)
    l_mean, l_std = compute_mean_std(l_list)
    w_mean, w_std = compute_mean_std(w_list)
    h_mean, h_std = compute_mean_std(h_list)
    return (l_mean, l_std), (w_mean, w_std), (h_mean, h_std)

def count_boxes_in_folder(folder_path):
    total_boxes = 0
    category_counter = defaultdict(int)
    category_pts = defaultdict(list)
    category_dims = defaultdict(list)

    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_boxes += len(data)
                            for item in data:
                                label = item.get("label", "Unknown")
                                subtype = item.get("subtype", "Unknown")
                                num_pts = item.get("num_lidar_pts", None)
                                lwh = item.get("lwh", None)

                                # ✅ 分类键：Vehicle 使用 subtype，其他用 label 本身
                                if label == "Vehicle":
                                    key = subtype
                                else:
                                    key = label

                                category_counter[key] += 1

                                if num_pts is not None:
                                    category_pts[key].append(num_pts)

                                if lwh and len(lwh) == 3:
                                    category_dims[key].append(lwh)

                except Exception as e:
                    print(f"读取出错: {json_path}, 错误: {e}")
    
    return total_boxes, category_counter, category_pts, category_dims

def merge_counters(counter_list):
    merged = defaultdict(int)
    for c in counter_list:
        for k, v in c.items():
            merged[k] += v
    return merged

def merge_label_pts(label_pts_list):
    merged = defaultdict(list)
    for label_pts in label_pts_list:
        for k, vlist in label_pts.items():
            merged[k].extend(vlist)
    return merged

# ✅ 主函数修改：合并并统计尺寸信息
def count_boxes_multiple_folders(root_folders):
    grand_total = 0
    all_counters = []
    all_pts = []
    all_dims = []

    for folder in root_folders:
        print(f"\n📁 正在统计文件夹: {folder}")
        total, counter, pts, dims = count_boxes_in_folder(folder)
        grand_total += total
        all_counters.append(counter)
        all_pts.append(pts)
        all_dims.append(dims)

        print(f"  📦 标注框总数: {total}")
        print("  📊 按类别统计:")
        for cat, count in counter.items():
            print(f"    {cat}: {count}")

    print("\n============================")
    print("📈 所有文件夹合计统计结果")
    print("============================")
    print(f"📦 总标注框数: {grand_total}")

    total_counts = merge_counters(all_counters)
    merged_pts = merge_label_pts(all_pts)
    merged_dims = merge_label_dims(all_dims)

    print("📊 合并后类别统计:")
    for cat, count in total_counts.items():
        print(f"  {cat}: {count}")

    print("\n📉 点数统计:")
    for cat, pts_list in merged_pts.items():
        mean, std = compute_mean_std(pts_list)
        print(f"  {cat}: 平均点数 = {mean:.2f}, 标准差 = {std:.2f}")

    print("\n📏 尺寸统计:")
    for cat, dims in merged_dims.items():
        (l_mean, l_std), (w_mean, w_std), (h_mean, h_std) = compute_lwh_stats(dims)
        print(f"  {cat}:")
        print(f"    长 l: 平均 = {l_mean:.2f}, 标准差 = {l_std:.2f}")
        print(f"    宽 w: 平均 = {w_mean:.2f}, 标准差 = {w_std:.2f}")
        print(f"    高 h: 平均 = {h_mean:.2f}, 标准差 = {h_std:.2f}")

def plot_histograms(pts_dict, title_prefix=""):
    for label, pts_list in pts_dict.items():
        if not pts_list:
            continue
        plt.figure(figsize=(8, 4))
        sns.histplot(pts_list, bins=50, kde=True)
        plt.title(f"{title_prefix}{label} - 点数分布 (num_lidar_pts)")
        plt.xlabel("num_lidar_pts")
        plt.ylabel("数量")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_mean_std_scatter(pts_dict, title):
    labels = []
    means = []
    stds = []
    for label, pts_list in pts_dict.items():
        if not pts_list:
            continue
        mean, std = compute_mean_std(pts_list)
        labels.append(label)
        means.append(mean)
        stds.append(std)

    plt.figure(figsize=(10, 6))
    plt.scatter(means, stds)

    for i, label in enumerate(labels):
        plt.annotate(label, (means[i], stds[i]), fontsize=8, alpha=0.7)

    plt.xlabel("平均点数")
    plt.ylabel("标准差")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    


def extract_classes_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    classes = set()
    for obj in data:
        label = obj.get("label")
        subtype = obj.get("subtype")
        if label == "Vehicle":
            if subtype and subtype != "None":
                classes.add(subtype)
        elif label in ["Pedestrian", "Cone"]:
            classes.add(label)
    return classes

def extract_classes_from_folder(folder):
    classes = set()
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                classes.update(extract_classes_from_json(json_path))
    return classes

def find_minimal_combination(root_dirs):
    # 每个路径对应的类别集合
    path_to_classes = {p: extract_classes_from_folder(p) for p in root_dirs}

    # 尝试组合，找到最小路径组合满足全类别
    for r in range(1, len(root_dirs) + 1):
        for combo in combinations(root_dirs, r):
            combined_classes = set()
            for path in combo:
                combined_classes.update(path_to_classes[path])
            if TARGET_CLASSES.issubset(combined_classes):
                return combo, combined_classes
    return None, set()


def print_minimal_combination_summary(root_dirs):
    # 寻找最小组合
    combo, found_classes = find_minimal_combination(root_dirs)

    print("✅ 最小路径组合：")
    for path in combo:
        print(" -", path)

    print("\n📋 类别覆盖：")
    for cls in sorted(found_classes):
        print(" -", cls)

if __name__ == '__main__':
    root_dirs = [
        # '/home/baojiali/Downloads/disk4/corage/label_ori/s1',
        # '/home/baojiali/Downloads/disk4/corage/label_ori/s2',
        # '/home/baojiali/Downloads/disk4/corage/label_ori/s3',
        # '/home/baojiali/Downloads/disk4/corage/label_ori/s5',
        # '/home/baojiali/Downloads/disk4/corage/label_ori/s6',
        # '/home/baojiali/Downloads/disk4/corage/label_ori/s7',
        # '/home/baojiali/Downloads/disk4/corage/label_ori/s9',
        # '/home/baojiali/Downloads/disk4/corage/label_ori/s11',
        
        # 'data/kl/v1.0-trainval/label/20250507',
        # 'data/kl/v1.0-trainval/label/20250521',
        # 'data/kl/v1.0-trainval/label/20250522',
        # 'data/kl/v1.0-trainval/label/20250515_20250516',
        # 'data/kl/v1.0-trainval/label/20250522_8',
        # 'data/kl/v1.0-trainval/label/20250523',
        # 'data/kl/v1.0-trainval/label/20250527',
        # 'data/kl/v1.0-trainval/label/20250604',
        # 'data/kl/v1.0-trainval/label/20250605',
        # 'data/kl/v1.0-trainval/label/20250606',
        # 'data/kl/v1.0-trainval/label/20250607',
        # 'data/kl/v1.0-trainval/label/20250609',
        # 'data/kl/v1.0-trainval/label/20250611',
        # 'data/kl/v1.0-trainval/label/20250612',
        # 'data/kl/v1.0-trainval/label/20250613',
        # 'data/kl/v1.0-trainval/label/20250618',
        # 'data/kl/v1.0-trainval/label/20250619',
        # 'data/kl/v1.0-trainval/label/20250620',
        # 'data/kl/v1.0-trainval/label/20250621',
        # 'data/kl/v1.0-trainval/label/20250624',
        # 'data/kl/v1.0-trainval/label/20250625',
        
        # 'data/kl/v1.0-trainval/label/20250627',
        # 'data/kl/v1.0-trainval/label/20250630',
        # 'data/kl/v1.0-trainval/label/20250702',
        # 'data/kl/v1.0-trainval/label/20250703',
        # 'data/kl/v1.0-trainval/label/20250705',
        
        # 'data/kl/v1.0-trainval/label/20250709',
        # 'data/kl/v1.0-trainval/label/20250709_chache',
        # 'data/kl/v1.0-trainval/label/20250710',
        
        # 'data/kl/v1.0-trainval/label/20250709_huichen',
        # 'data/kl/v1.0-trainval/label/20250709_huichen2',
        # 'data/kl/v1.0-trainval/label/20250710_2',
        # 'data/kl/v1.0-trainval/label/20250711_huichen',
        # 'data/kl/v1.0-trainval/label/20250716_duigaoji_luntaidiao',
        
        # 'data/kl/v1.0-trainval/label/20250715',
        # 'data/kl/v1.0-trainval/label/20250716_duigaoji_paoquan',
        # 'data/kl/v1.0-trainval/label/20250719_zhuangxiang_zhengmiandiao',
        
        # 'data/kl/v1.0-trainval/label/20250719_chache',
        # 'data/kl/v1.0-trainval/label/20250721_baiche_henxiang',
        
        # 'data/kl/v1.0-trainval/label/20250722_dark_test1',
        # 'data/kl/v1.0-trainval/label/20250722_dark_test2',
        # 'data/kl/v1.0-trainval/label/20250722_dark_test3',
        
        # 'data/kl/v1.0-trainval/label/20250728',
        # 'data/kl/v1.0-trainval/label/20250729',
        
        # 'data/kl/v1.0-trainval/label/20250730',
        # 'data/kl/v1.0-trainval/label/20250731',
        
        # 'data/kl/v1.0-trainval/label/20250801',
        # 'data/kl/v1.0-trainval/label/20250802',
        # 'data/kl/v1.0-trainval/label/20250804',

        # 'data/kl/v1.0-trainval/label/20250806',
        # 'data/kl/v1.0-trainval/label/20250807',
        
        # 'data/kl/v1.0-trainval/label/20250814',
        # 'data/kl/v1.0-trainval/label/20250815',
        # 'data/kl/v1.0-trainval/label/20250816',
        # 'data/kl/v1.0-trainval/label/20250818',

        # 'data/kl/v1.0-trainval/label/20250820',
        # 'data/kl/v1.0-trainval/label/20250821',
        # 'data/kl/v1.0-trainval/label/20250822',
        # 'data/kl/v1.0-trainval/label/20250826',
        # 'data/kl/v1.0-trainval/label/20250827',
        # 'data/kl/v1.0-trainval/label/20250828',
        
        # 'data/kl/v1.0-trainval/label/20250829',
        # 'data/kl/v1.0-trainval/label/20250830',
        # 'data/kl/v1.0-trainval/label/20250901_002',
        # 'data/kl/v1.0-trainval/label/20250901_005',
        # 'data/kl/v1.0-trainval/label/20250902',
        # 'data/kl/v1.0-trainval/label/20250903',
        # 'data/kl/v1.0-trainval/label/20250904',
        # 'data/kl/v1.0-trainval/label/20250912',
        # 'data/kl/v1.0-trainval/label/20250913',
        
        # 'data/jinan/v1.0-trainval/label/20251009_005',
        # 'data/jinan/v1.0-trainval/label/20251010_006',
        # 'data/jinan/v1.0-trainval/label/20251011_006',
        # 'data/jinan/v1.0-trainval/label/20251013_006',
        
        # 'data/jinan/v1.0-trainval/label/20251025_002',
        # 'data/jinan/v1.0-trainval/label/20251025_005',
        # 'data/jinan/v1.0-trainval/label/20251028_005',
        # 'data/jinan/v1.0-trainval/label/20251029_002',
        # 'data/jinan/v1.0-trainval/label/20251029_006',
        
        # 'data/jinan/v1.0-trainval/label/20250929_006',
        # 'data/jinan/v1.0-trainval/label/20251106_005',
        # 'data/jinan/v1.0-trainval/label/20251106_006_jianguanchangnei',
        # 'data/jinan/v1.0-trainval/label/20251106_006_xingren_sanlun',
        # 'data/jinan/v1.0-trainval/label/20251108_006',
        # 'data/jinan/v1.0-trainval/label/20251110_005',
        # 'data/jinan/v1.0-trainval/label/20251112_005',
        # 'data/jinan/v1.0-trainval/label/20251113_005',
        # 'data/jinan/v1.0-trainval/label/20251114_005',
        # 'data/jinan/v1.0-trainval/label/20251119_005',
        
        
        'data/kl_8/v1.0-trainval/label/20251027_007',
        'data/kl_8/v1.0-trainval/label/20251028_007',
        'data/kl_8/v1.0-trainval/label/20251029_007',
        'data/kl_8/v1.0-trainval/label/20251031_007',
        'data/kl_8/v1.0-trainval/label/20251105_007',
        'data/kl_8/v1.0-trainval/label/20251106_007',
        'data/kl_8/v1.0-trainval/label/20251108_007',
        'data/kl_8/v1.0-trainval/label/20251110_007',
        'data/kl_8/v1.0-trainval/label/20251112_007',
        'data/kl_8/v1.0-trainval/label/20251113_007_chache',
        'data/kl_8/v1.0-trainval/label/20251114_007',
        'data/kl_8/v1.0-trainval/label/20251118_007',
        'data/kl_8/v1.0-trainval/label/20251119_007',
    ]
    
    

    
    count_boxes_multiple_folders(root_dirs)
    
    # print_minimal_combination_summary(root_dirs)
