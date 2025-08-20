import pandas as pd
import matplotlib.pyplot as plt


def compare_models(csv_paths, labels):
    """对比不同模型的CSV记录"""
    plt.figure(figsize=(10, 6))
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        plt.plot(df['epoch'], df['accuracy'], label=f"{label} (Train)")
        plt.plot(df['epoch'], df['val_accuracy'], '--', label=f"{label} (Val)")

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("results/accuracy_comparison.png")


if __name__ == "__main__":
    # 示例：对比经典模型和量子混合模型
    compare_models(
        csv_paths=["results/metrics.csv", "results/classical_metrics.csv"],
        labels=["Quantum Hybrid", "Classical"]
    )