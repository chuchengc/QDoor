from qiskit import *
from qiskit import Aer, QuantumCircuit

import numpy as np
from bqskit.ir.gates import CNOTGate, RZGate, RXGate, CZGate, SXGate, U8Gate, U3Gate, U2Gate, U1Gate
from bqskit import MachineModel
import torch
import pennylane as qml
from qiskit.tools.visualization import plot_histogram
import pennylane_qiskit
from bqskit.ext import qiskit_to_bqskit
from bqskit import compile


synthesis_epsilon = 1e-2


num_inputs = 8
params = np.ones((8, 8))*3
param_dic = torch.load("./model/init_whole8.pt")
params = param_dic["qlayer.w"].detach().numpy()


qr = QuantumRegister(8, "q")
circuit = QuantumCircuit(qr)

##########################################circuit 1###########################################################
for i in range(num_inputs):
    circuit.rx(params[0*8 + i], qr[i])
    circuit.ry(params[1*8 + i], qr[i])
    circuit.rz(params[2*8 + i], qr[i])

circuit.crx(params[3*8 + 0], qr[0], qr[1])
circuit.crx(params[3*8 + 1], qr[2], qr[3])
circuit.crx(params[3*8 + 2], qr[4], qr[5])
circuit.crx(params[3*8 + 3], qr[6], qr[7])

################################################################################################################


circuit = qiskit_to_bqskit(circuit)

print("Circuit Statistics")
print("Gate Counts:", circuit.gate_counts)
print("Logical Connectivity:", circuit.coupling_graph)


gate_set = {CNOTGate(), U2Gate()} 
model = MachineModel(circuit.num_qudits, gate_set=gate_set)

out_circuit = compile(circuit, model=model, optimization_level=3, synthesis_epsilon=synthesis_epsilon)

# Print new statistics
print("Compiled Circuit Statistics")
print("Gate Counts:", out_circuit.gate_counts)
print("Logical Connectivity:", out_circuit.coupling_graph)

out_circuit.save('./qasm/com_circuit.qasm')

