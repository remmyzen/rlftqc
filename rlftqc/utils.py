from qiskit import QuantumCircuit

def convert_stim_to_qiskit_circuit(encoding_circuit):
    """ Convert stim circuit to Qiskit circuit.
    
    Args:
        encoding_circuit: Encoding circuit in Stim format. 
    """
    qc = QuantumCircuit(encoding_circuit.num_qubits)

    for circ in encoding_circuit:
        # Split the text
        spl = str(circ).strip().split(' ')
        gate_name = spl[0]
        # two qubit gates
        if gate_name == 'CX':
            for ii in range(0, (len(spl) - 1), 2):
                qc.cx(int(spl[ii+1]), int(spl[ii+2]))
        elif gate_name == 'CZ':
            for ii in range(0, (len(spl) - 1), 2):
                qc.cz(int(spl[ii+1]), int(spl[ii+2]))
        # one qubit gate
        elif gate_name == 'H':
            for ii in range(1, len(spl)):
                qc.h(int(spl[ii]))
        elif gate_name == 'S':
            for ii in range(1, len(spl)):
                qc.s(int(spl[ii]))
        elif gate_name == 'SQRT_X':
            for ii in range(1, len(spl)):
                qc.sx(int(spl[ii]))
        elif gate_name == 'X':
            for ii in range(1, len(spl)):
                qc.x(int(spl[ii]))
    return qc

