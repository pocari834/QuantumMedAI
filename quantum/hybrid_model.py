import numpy as np
import pennylane as qml
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from transformers import TFBertModel

# --------------------------
# 量子电路定义 (PennyLane)
# --------------------------
n_qubits = 4  # 量子比特数
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    """可训练量子电路"""
    # 编码经典数据到量子态 (角度编码)
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)

    # 可训练量子层
    for i in range(n_qubits):
        qml.Rot(*weights[i], wires=i)

    # 纠缠层
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])

    # 测量期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# --------------------------
# 混合模型构建
# --------------------------
def build_hybrid_model(image_size=(128, 128, 1), text_max_length=128):
    """构建量子-经典多模态模型"""

    # ------------------
    # 1. 图像分支 (量子)
    # ------------------
    image_input = Input(shape=image_size, name="image_input")

    # 经典CNN预处理 (降维到n_qubits维度)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(image_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu')(x)  # 输出通道=量子比特数
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # 量子层 (通过KerasLayer包装)
    weight_shapes = {"weights": (n_qubits, 3)}  # 每个量子比特3个旋转参数
    quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
    quantum_features = quantum_layer(x)

    # ------------------
    # 2. 文本分支 (经典BERT)
    # ------------------
    text_input = Input(shape=(text_max_length,), dtype=tf.int32, name="text_input")
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    text_features = bert_model(text_input).last_hidden_state
    text_features = GlobalAveragePooling1D()(text_features)

    # ------------------
    # 3. 多模态融合
    # ------------------
    combined = Concatenate()([quantum_features, text_features])

    # 分类头
    output = Dense(2, activation='softmax', name="output")(combined)  # 假设二分类

    return Model(inputs=[image_input, text_input], outputs=output)


# --------------------------
# 辅助函数
# --------------------------
def load_pretrained_bert():
    """加载预训练BERT模型和tokenizer"""
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


if __name__ == "__main__":
    # 测试模型构建
    model = build_hybrid_model()
    model.summary()
    tf.keras.utils.plot_model(model, "hybrid_model.png", show_shapes=True)