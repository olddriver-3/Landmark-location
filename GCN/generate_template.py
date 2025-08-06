import os
import glob

def compute_average_txt(input_folder, output_filename="template.txt"):
    # 获取文件夹中所有的txt文件（排除template.txt）
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    txt_files = [f for f in txt_files if os.path.basename(f) != output_filename]
    
    if not txt_files:
        print("没有找到可处理的txt文件")
        return
    
    # 初始化数据存储
    data = []
    line_counts = []
    
    # 读取所有文件的数据
    for file_path in txt_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            line_counts.append(len(lines))
            file_data = []
            for line in lines:
                line = line.strip()
                if line:
                    x, y = map(int, line.split(','))
                    file_data.append((x, y))
            data.append(file_data)
    
    # 检查所有文件的行数是否一致
    if len(set(line_counts)) > 1:
        print("警告：不同txt文件的行数不一致，将按照最小行数计算平均值")
        min_lines = min(line_counts)
        data = [d[:min_lines] for d in data]
    
    # 计算平均值
    averaged_data = []
    for i in range(len(data[0])):
        sum_x = 0
        sum_y = 0
        count = 0
        for file_data in data:
            if i < len(file_data):
                x, y = file_data[i]
                sum_x += x
                sum_y += y
                count += 1
        avg_x = round(sum_x / count)
        avg_y = round(sum_y / count)
        averaged_data.append(f"{avg_x},{avg_y}")
    
    # 写入结果到template.txt
    output_path = os.path.join(input_folder, output_filename)
    with open(output_path, 'w') as f:
        f.write('\n'.join(averaged_data))
    
    print(f"平均值已计算并保存到 {output_path}")

# 使用示例
input_folder = "./converted_labels"  # 当前文件夹，可以替换为你的文件夹路径
compute_average_txt(input_folder)
