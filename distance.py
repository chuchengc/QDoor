import numpy as np
from qiskit import *
from qiskit import Aer, QuantumCircuit
import torch
from bqskit.ext import qiskit_to_bqskit
from bqskit import Circuit



def circuit_init(params):
    qubit_num = 8
    qr = QuantumRegister(qubit_num, "q")
    circuit_qiskit = QuantumCircuit(qr)

    for i in range(qubit_num):
        circuit_qiskit.rx(params[0*8 + i], qr[i])
        circuit_qiskit.ry(params[1*8 + i], qr[i])
        circuit_qiskit.rz(params[2*8 + i], qr[i])


    circuit_qiskit.crx(params[3*8 + 0], qr[0], qr[1])
    circuit_qiskit.crx(params[3*8 + 1], qr[2], qr[3])
    circuit_qiskit.crx(params[3*8 + 2], qr[4], qr[5])
    circuit_qiskit.crx(params[3*8 + 3], qr[6], qr[7])

    return circuit_qiskit

####################################################################################
param_dic = torch.load("./model/init_whole8.pt")
params = param_dic["qlayer.w"].detach().numpy()

compile_circuit = Circuit.from_file("./qasm/circuit.qasm")
original_circuit = circuit_init(params)

compile_matrix = compile_circuit.get_unitary()
original_matrix = qiskit_to_bqskit(original_circuit).get_unitary()

A = original_matrix
B = compile_matrix.numpy


epsilon = np.abs(1 - np.abs(np.sum(np.multiply(A,np.conj(B)))) / A.shape[0])
print(epsilon)



