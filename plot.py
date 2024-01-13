import json,os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt


def plot_results(file_path='./experiments/result_tree.json', save_path='./experiments/plot.png'):
    # 读取结果
    with open(file_path, 'r') as file:
        results = json.load(file)

    # 准备绘图数据
    dimensions = list(results.keys())
    accuracies = [results[dim] for dim in dimensions]

    # 绘制图形
    plt.clf() 
    plt.plot(dimensions, accuracies, marker='o')
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs PCA Dimensions')

    # 保存图形
    plt.savefig(save_path)

if __name__ == '__main__':
    os.makedirs('./figure/',exist_ok=True)
    plot_results('./experiments/DecisionTree.json',save_path='./figure/DecisionTree.png')
    plot_results('./experiments/svm.json',save_path='./figure/svm.png')
