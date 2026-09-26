# IBM Quantum Devices

Resulting circuit from RL with IBM Quantum devices with the native gate set (SQRT_X, X, S, and CNOT gates).

Different codes:

1. $|0\rangle$ of $[[5,1,3]]$ perfect code in IBMQ Manila [5-1-3-manila](5-1-3-manila)
2. $|0\rangle$ of $[[5,1,3]]$ perfect code in IBMQ Lima [5-1-3-lima](5-1-3-lima)
3. $|0\rangle$ of $[[7,1,3]]$ Steane code in IBMQ Jakarta [7-1-3-jakarta](7-1-3-jakarta)
4. $|+\rangle$ of $[[9,1,3]]$ Shor code in IBMQ Guadalupe [9-1-3-guadalupe](9-1-3-guadalupe)
5. $|0\rangle$ of $[[15,1,3]]$ Reed-Muller / 3D color code in IBMQ Tokyo [15-1-3-tokyo](15-1-3-tokyo)

The qubit placement is given in `qubit_place.txt` according to the notation in the Qiskit library. If the qubit placement is given as $a,b,c,\dots$, then it means that qubit $0$ ($q_0$) in the circuit is placed in qubit $a$ on the device, qubit $1$ ($q_1$)  is placed in qubit $b$ on the device, and so on. 
