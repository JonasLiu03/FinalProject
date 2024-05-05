import matplotlib.pyplot as plt
import re  # 正则表达式库，用于从文件名中提取数字


def read_distances(file_path):
    distances = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(": LPIPS distance = ")
            checkpoint = parts[0].strip()
            checkpoint_number = ''.join(re.findall(r'\d+', checkpoint))
            distance = float(parts[1].strip())
            distances[checkpoint_number] = distance
    return distances


def plot_distances(distances):

    sorted_checkpoints = sorted(distances.items(), key=lambda x: int(x[0]))
    checkpoints, lpips_values = zip(*sorted_checkpoints)
    plt.figure(figsize=(10, 5))
    plt.scatter(checkpoints, lpips_values, color='skyblue')
    plt.plot(checkpoints, lpips_values, color='lightgray', linestyle='--')  # 添加连线以提高可读性
    plt.text(checkpoints[0], lpips_values[0], f'{lpips_values[0]:.3f}', fontsize=12, verticalalignment='bottom')
    plt.text(checkpoints[-1], lpips_values[-1], f'{lpips_values[-1]:.3f}', fontsize=12, verticalalignment='bottom')
    plt.ylabel('LPIPS Distance')
    plt.xlabel('Model Checkpoint Number')
    plt.title('LPIPS Distance for Different Model Checkpoints')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# 主函数
def main():
    file_path = 'lpips_distances.txt'
    distances = read_distances(file_path)
    plot_distances(distances)

if __name__ == "__main__":
    main()
