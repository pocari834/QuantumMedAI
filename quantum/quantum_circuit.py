from pennylane import Rot, RY, PauliZ, device, expval, qnode, CNOT

n_qubits = 4
dev = device("default.qubit", wires=n_qubits)


@qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    # 数据编码层
    for i in range(n_qubits):
        RY(inputs[i] * np.pi, wires=i)

    # 可训练量子层
    for i in range(n_qubits):
        Rot(*weights[i], wires=i)

    # 纠缠层
    CNOT(wires=[0, 1])
    CNOT(wires=[2, 3])

    return [expval(PauliZ(i)) for i in range(n_qubits)]