# 2D Grid

Resulting circuit from RL in 2D grid connectivity based on Google Sycamore with the standard gate set (H, S, and CNOT gate) + CZ gate.

Different codes:
1. The $|0\rangle$ state of the $[[7,1,3]]$ Steane code [7-1-3](7-1-3)
2. The $|+\rangle$ state of the $[[9,1,3]]$ Shor code [9-1-3-shor](9-1-3-shor)
2. The $|0\rangle$ state of the $[[9,1,3]]$ Surface-17 code [9-1-3-surface](9-1-3-surface)

The qubit placement is given in `qubit_place.txt` according to the notation in the Qiskit library. See the `connectivity.png` for the qubit placement numbers.

If the qubit placement is given as $a,b,c,\dots$, then it means that qubit $0$ ($q_0$) in the circuit is placed in qubit $a$ on the device, qubit $1$ ($q_1$)  is placed in qubit $b$ on the device, and so on. 
