# IBM Quantum Device

Resulting circuits from RL with the standard gate set (H, S, and CNOT gates) + CZ gate.

Different codes:

1. The $|1\rangle$ state of the $[[5,1,3]]$ perfect code [5-1-3-tokyo](5-1-3)
2. The $|0\rangle$ state of the $[[7,1,3]]$ Steane code [7-1-3-guadalupe](7-1-3)

The qubit placement is given in `qubit_place.txt` according to the notation in the Qiskit library. If the qubit placement is given as $a,b,c,\dots$, then it means that qubit $0$ ($q_0$) in the circuit is placed in qubit $a$ on the device, qubit $1$ ($q_1$)  is placed in qubit $b$ on the device, and so on. 
