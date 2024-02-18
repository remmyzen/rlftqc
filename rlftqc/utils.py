from qiskit import QuantumCircuit
import stim

def convert_stim_to_qiskit_circuit(encoding_circuit, num_ancillas=0):
    """ Convert stim circuit to Qiskit circuit.
    
    Args:
        encoding_circuit: Encoding circuit in Stim format. 
        num_ancillas: Number of ancillas is needed for drawing verification circuit synthesis task

    Returns:
        Qiskit QuantumCircuit object of the same circuit
    """
    qc = QuantumCircuit(encoding_circuit.num_qubits + num_ancillas)

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

def convert_qiskit_to_stim_circuit(encoding_circuit):
    """ Convert stim circuit to Qiskit circuit.
    
    Args:
        encoding_circuit: Encoding circuit in qiskit QuantumCircuit format. 

    Returns:
        Stim Circuit object of the same circuit
    """

    circ = stim.Circuit()

    for instr, qargs, _ in encoding_circuit.data:
        gate_name = instr.name
        qubit_indices = [q.index for q in qargs]
        # two qubit gates
        if gate_name == 'cx':
            circ.append(gate_name, qubit_indices)
        elif gate_name == 'cz':
            circ.append(gate_name, qubit_indices)
        # one qubit gate
        elif gate_name == 'h':
            circ.append(gate_name, qubit_indices)
        elif gate_name == 's':
            circ.append(gate_name, qubit_indices)
        elif gate_name == 'sx':
            circ.append("sqrt_x", qubit_indices)
        elif gate_name == 'x':
            circ.append(gate_name, qubit_indices)
        else:
            # does not handle other clifford gates and non-clifford gate
            raise NotImplementedError("%s gate is not yet supported", gate_name)

    return circ

def is_css_code(stabilizer_generators):
    """ Check if the given generators is a CSS code (every generators only contains all X or Z).

    This is used to give hints for synthesizing the verification circuit.

    Args:
        stabilizer_generators (list(str)): The list of stabilizer generators as a list of string.

    Returns:
        If the code is CSS or not.
    """
    is_css = True
    for generator in stabilizer_generators:
        # Check if Y is in the generator or X and Z is in the generator.
        if 'y' in generator.lower() or ('x' in generator.lower() and 'z' in generator.lower()):
            is_css = False
            break
    return is_css