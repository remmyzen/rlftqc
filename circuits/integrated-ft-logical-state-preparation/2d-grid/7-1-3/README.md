# 2D Grid Steane code

Resulting circuit from RL in 2D grid connectivity based on Google Sycamore with the standard gate set (H, S, and CNOT gate) + CZ gate for $[[7,1,3]]$ Steane code.

We take qubits 2,3,4,7,8,9,14,15,16 from the `../connectivity.png`. The different folders shows the position of the two flag/ancilla qubits.

The qubit placement is given in `qubit_place.txt` according to the notation in the Qiskit library. 

If the qubit placement is given as $a,b,c,\dots$, then it means that qubit $0$ ($q_0$) in the circuit is placed in qubit $a$ on the device, qubit $1$ ($q_1$)  is placed in qubit $b$ on the device, and so on. 
