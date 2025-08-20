import csv
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append(r"C:\Users\sihuuhuang\PycharmProjects\pythonProject4")  # 添加项目根目录
from quantum.hybrid_model import build_hybrid_model
from preprocessing.data_loader import OpenIDataset

# 初始化结果目录
import os

os.makedirs("results/accuracy_plots", exist_ok=True)
os.makedirs("results/quantum_circuits", exist_ok=True)


# 训练记录函数
def log_metrics(epoch, logs, csv_path="results/metrics.csv"):
    row = {
        'epoch': epoch,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'loss': logs['loss'],
        'accuracy': logs.get('accuracy', None),
        'val_loss': logs.get('val_loss', None),
        'val_accuracy': logs.get('val_accuracy', None)
    }

    # 写入CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# 可视化量子电路
def save_quantum_circuit():
    qml.drawer.use_style('black_white')
    fig, ax = qml.draw_mpl(quantum_circuit)(inputs=[0.1] * n_qubits, weights=[[0.1, 0.2, 0.3]] * n_qubits)
    fig.savefig(f"results/quantum_circuits/circuit_{datetime.now().strftime('%Y%m%d')}.png", dpi=300)


# 主训练逻辑
def train():
    model = build_hybrid_model()
    dataset = OpenIDataset("data/metadata.csv")

    # 回调函数
    class MetricsLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_metrics(epoch, logs)
            save_accuracy_plot()

    # 训练
    history = model.fit(
        dataset,
        epochs=10,
        callbacks=[MetricsLogger()]
    )

    # 保存最终准确度曲线
    save_accuracy_plot(history)


# 生成准确度曲线
def save_accuracy_plot(history=None, filename=None):
    if history:
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f"results/accuracy_plots/accuracy_{datetime.now().strftime('%Y%m%d')}.png")
        plt.close()


if __name__ == "__main__":
    save_quantum_circuit()  # 初始量子电路图
    train()