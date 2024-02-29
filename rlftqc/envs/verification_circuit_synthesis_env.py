from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
from flax import struct
import chex
from inspect import signature
from typing import Tuple, Optional
from qiskit import QuantumCircuit
import stim
from rlftqc.simulators import PauliString, TableauSimulator, CliffordGates
from rlftqc.utils import convert_qiskit_to_stim_circuit, is_css_code
import itertools

@struct.dataclass
class EnvState:
    """ This class will contain the state of the environment."""
    tableau: jnp.array
    signs: jnp.array
    propagated_error: jnp.array
    previous_flagged_errors: float
    previous_product_ancilla: float
    previous_distance: float
    number_of_errors: int
    time: int

@struct.dataclass
class EnvParams:
    """ This class will contain the default parameters of the environment."""
    max_steps: int = 10

class VerificationCircuitSynthesisEnv(environment.Environment):
    """Environment for the verification circuit synthesis task.

    Args:
        encoding_circuit: The logical state preparation circuit in Stim (stim.Circuit) or Qiskit (qiskit.circuit.QuantumCircuit).
        num_ancillas (int, optional): The number of flag qubits in the verification circuit. Default: 1.
        distance (int, optional): The distance of the code of the logical state. Currently only supports distance 3.
        gates (list(CliffordGates), optional): List of clifford gates to prepare the verification circuit. Default: H, CX, CZ. 
        graph (list(tuple), optional): Graph of the qubit connectivity. Default: all-to-all qubit connectivity.
        max_steps (int, optional): The number of maximum gates to be applied in the circuit. Default: 10
        threshold (float, optional): The complementary distance threshold to indicates success. Default: 0.99
        mul_errors_with_generators (boolean, optional): If set to true will multiply errors with the generators to reduce the weight. Default: True.
        mul_errors_with_S (boolean, optional): If set to true will multiply errors with the S to reduce the weight. Default: False. Useful for non-CSS codes but will generate S which is exponential.
        ignore_x_errors (boolean, optional): If set to true will ignore any x errors. Useful for preparing |+> or |-> of CSS codes. Default: false.
        ignore_y_errors (boolean, optional): If set to true will ignore any y errors. Useful for preparing |+i> or |-i> of CSS codes. Default: false.
        ignore_z_errors (boolean, optional): If set to true will ignore any z errors. Useful for preparing |0> or |1> of CSS codes. Default: false.
        weight_distance (float, optional): The weight of the distance reward (\mu_d in the paper). Default: 1..
        weight_flag (float, optional): The weight of the flag reward, (\mu_f in the paper). Default: number of qubits.
        weight_ancillas (float, optional): The weight of the product ancilla reward, (\mu_p in the paper). Default: number_of_qubits // 2.
        ancilla_target_only (boolean, optional): If set to True, ancilla is only going to be the target of cnot and not control (sometimes useful for faster convergence). Default: False.
        ancilla_control_only (boolean, optional): If set to True, ancilla is only going to be the control of cnot and not target (sometimes useful for faster convergence). Default: False.
        gates_between_ancilla (boolean, optional): If set to True, this allows two qubit gates between ancillas. Default: True.
        gates_between_data (boolean, optional): If set to True, this allows one and two qubit gates in the data qubits. Default: False.
        group_ancillas (boolean, optional): If set to True, this will group ancilla into two. Useful to replicate the protocol in Chamberland and Chao original paper. For example: If there are 4 flag qubits, there will be no two-qubit gates between flag qubits 1,2 and 3,4. 
        plus_ancilla_position (list(int), optional): Initialize flag qubits given in the list as plus state and will measure in the X basis.
            This is useful for non-CSS codes.
    """
    
    def __init__(self,
        encoding_circuit,
        num_ancillas = 1,     
        distance = 3,       
        gates=None,   
        graph=None,    
        max_steps = 10,
        threshold = 0.99999,                 
        mul_errors_with_generators = True,
        mul_errors_with_S = False,
        ignore_x_errors = False,
        ignore_y_errors = False,
        ignore_z_errors = False,
        weight_distance = None,
        weight_flag = None,
        weight_ancillas = None,
        ancilla_target_only = False,
        ancilla_control_only = False, 
        gates_between_ancilla = True,
        gates_between_data = False, 
        group_ancillas = False,  
        plus_ancilla_position = []
        ):
        """Initialize a verification circuit synthesis environment.
        """
        super().__init__()
    
        ## Process encoding circuit

        if isinstance(encoding_circuit, QuantumCircuit):
            self.encoding_circuit = convert_qiskit_to_stim_circuit(encoding_circuit)
        elif isinstance(encoding_circuit, stim.Circuit):
            self.encoding_circuit = encoding_circuit
        else:
            raise NotImplementedError("Only stim.Circuit or qiskit.QuantumCircuit object is supported.")
        


        self.n_qubits_physical_encoding = self.encoding_circuit.num_qubits
        self.num_ancillas = num_ancillas
        self.n_qubits_physical = self.n_qubits_physical_encoding + self.num_ancillas

        self.encoding_tableau = stim.Tableau(self.n_qubits_physical_encoding).from_circuit(self.encoding_circuit)

        ## Check for CSS
        self.is_css = is_css_code([str(self.encoding_tableau.z_output(ii)) for ii in range(self.n_qubits_physical_encoding)])

        ## Process gates
        self.gates = gates
        if self.gates is None:
            clifford_gates = CliffordGates(self.n_qubits_physical)
            ## Initialize with the standard gate set + CZ minus S for non-CSS
            if self.is_css:
                print("Code is CSS, but gate set is not specified using H and CX as the default gate set.")
                self.gates = [clifford_gates.h, clifford_gates.cx] 
            else:               
                print("Code is non-CSS, but gate set is not specified using H, CX, and CZ as the default gate set.")
                self.gates = [clifford_gates.h, clifford_gates.cz, clifford_gates.cx] 

        ## Create qubit connectivity graph
        self.graph = graph
        if self.graph is None:
            self.graph = []
            ## Fully connected
            for ii in range(self.n_qubits_physical):
                for jj in range(ii+1, self.n_qubits_physical):
                    self.graph.append((ii,jj))
                    self.graph.append((jj,ii))


        self.threshold = threshold
        self.max_steps = max_steps
        self.distance = distance


        self.mul_errors_with_generators = mul_errors_with_generators ## multiply error with generators might reduce number of errors but make the process slower
        self.mul_errors_with_S = mul_errors_with_S ## multiply error with S


        if not self.is_css and not self.mul_errors_with_S:
            print("Code is non-CSS, multiply errors with S is set to True. It might need a big memory for bigger codes.")
            self.mul_errors_with_S = True
        
        if self.mul_errors_with_S and self.mul_errors_with_generators:
            self.mul_errors_with_generators = False
            
        self.weight_flag = weight_flag

        if self.weight_flag is None:
            self.weight_flag = self.n_qubits_physical_encoding

        self.weight_ancillas = weight_ancillas
        if self.weight_ancillas is None:
            self.weight_ancillas = self.n_qubits_physical_encoding // 2

        self.weight_distance = weight_distance
        if self.weight_distance is None:
            self.weight_distance = 1.

        self.ignore_x_errors = ignore_x_errors
        self.ignore_y_errors = ignore_y_errors
        self.ignore_z_errors = ignore_z_errors   

        if self.is_css and not self.ignore_x_errors and not self.ignore_y_errors and not self.ignore_z_errors:
            print("WARNING! Code is CSS but no errors are ignored. You might want to set to ignore some error depending on the logical state that you are preparing. For example: For zero-logical, then Z errors can be ignored and add ignore_z_errors = True as an argument.")

        self.ancilla_target_only = ancilla_target_only
        self.ancilla_control_only = ancilla_control_only
        self.gates_between_ancilla = gates_between_ancilla
        self.gates_between_data = gates_between_data

        self.group_ancillas = group_ancillas
        self.plus_ancilla_position = jnp.array(plus_ancilla_position, dtype=jnp.int32)
        self.zero_ancilla_position = jnp.delete(jnp.arange(self.n_qubits_physical_encoding, self.n_qubits_physical_encoding+self.num_ancillas),  self.plus_ancilla_position  - self.n_qubits_physical_encoding)
        ### Process encoding tableau
        self.encoding_tableau_with_ancilla, self.encoding_tableau_signs = self.stim_tableau_to_numpy(self.encoding_tableau, num_ancillas=self.num_ancillas)
        self.initial_tableau_with_ancillas = TableauSimulator(self.n_qubits_physical_encoding, initial_tableau=self.encoding_tableau_with_ancilla, initial_sign =self.encoding_tableau_signs)
        
        generators = [str(self.encoding_tableau.z_output(ii))[1:] + ("I" * self.num_ancillas) for ii in range(self.n_qubits_physical_encoding)] 
        generators_signs = [0 if self.encoding_tableau.z_output(ii).sign == 1 else 1 for ii in range(self.n_qubits_physical_encoding) ] 
        
        self.encoding_generators = PauliString(generators)

        self.target_tableau = PauliString([gen[:self.n_qubits_physical_encoding] for gen in generators])        
        self.target_check_matrix_unflatten = jnp.array(self.target_tableau.to_numpy())

        canon_target_check_matrix, canon_target_sign = self.canonical_stabilizers(self.target_check_matrix_unflatten, jnp.array(generators_signs) * 2)
        self.target_check_matrix = jnp.append(canon_target_check_matrix.flatten(), canon_target_sign // 2)

        ## Only generate group if need to mul error with S
        if self.mul_errors_with_S:
            self.encoding_group = self.encoding_generators.generate_stabilizer_group()
        
        ## Get propagated error from the encoding circuit
        # Generate all two qubit errors except II...I
        self.two_qubit_errors = list(itertools.product('IXYZ', repeat=2))[1:]  
        self.initial_propagated_errors = self.get_propagated_errors()    

        self.number_of_initial_errors = self.initial_propagated_errors.batch_size

        ## Append the initial propagated errors with III..III (15 (number of errors) * max_steps) that will be filled with gate errors during step since Jax cannot handle dynamic array
        self.initial_propagated_errors = self.initial_propagated_errors.append(PauliString(self.initial_propagated_errors.num_qubits, 15 * self.max_steps))

        ## Generate action matrix
        self.actions = self.action_matrix()

        ## Generate observation
        self.obs_shape  =  self.get_observation(self.initial_tableau_with_ancillas.current_tableau[0]).flatten().shape

    def stim_tableau_to_numpy(self, stim_tableau, num_ancillas = 0):
        ''' Convert stim tableau to proper numpy tableau for our simulator.

        Args:
            stim_tableau: The input stim Tableau
            num_ancillas (int, optional): number of ancillas. Default: 0

        Returns:
            Check matrix and its signs in numpy format.
        '''

        check_mat_list = stim_tableau.to_numpy() 
        check_mat = jnp.block([[jnp.pad(check_mat_list[0], (0, num_ancillas)), jnp.pad(check_mat_list[1], (0, num_ancillas))],[jnp.pad(check_mat_list[2], (0, num_ancillas)), jnp.pad(check_mat_list[3], (0, num_ancillas))]]).astype(jnp.uint8)
        ## Set ancilla tablau
        for ii in range(num_ancillas):
            check_mat = check_mat.at[2 * self.n_qubits_physical - ii - 1, 2 * self.n_qubits_physical -  ii - 1].set(1)
        signs = list(check_mat_list[4]) + ([0] * num_ancillas) + list(check_mat_list[5]) + ([0] * num_ancillas)
        return check_mat, jnp.array(signs)
   

    def get_observation(self, tableau):
        """ Extract the check matrix for the observation of the RL agent.
        
        Args:
            tableau: check matrix of the tableau.

        Returns:
            Returns the stabilizer part of the tableau and ignore the destabilizers.
        """  
        check_mat = tableau[self.n_qubits_physical:].astype(jnp.uint8)
        return check_mat

    def get_observation_without_ancilla(self, tableau):
        """ Extract the check matrix without the ancilla to calculate distance with target.

        Args:
            tableau: check matrix of the tableau.

        Returns:
            Returns the stabilizer part of the tableau without ancilla
        """
        ## Remove the ancilla
        check_mat_x = tableau[self.n_qubits_physical:self.n_qubits_physical + self.n_qubits_physical_encoding, :self.n_qubits_physical_encoding]
        check_mat_z = tableau[self.n_qubits_physical:self.n_qubits_physical + self.n_qubits_physical_encoding, self.n_qubits_physical: self.n_qubits_physical + self.n_qubits_physical_encoding]
     
        check_mat = jnp.block([check_mat_x, check_mat_z]).astype(jnp.uint8)
        return check_mat   

    def split_stim_circuit(self):
        """ Split gate from Stim encoding circuit to enable insertion of error to propagate.
        e.g. H 0 1 into H 0, H 1.
        
        Returns:
            Splitted Stim circuit.
        """        
        circuits_string = []
        for circ in self.encoding_circuit:
            # Split the text
            spl = str(circ).strip().split(' ')
            gate_name = spl[0]
            # two qubit gates
            if gate_name == 'CX':
                for ii in range(0, (len(spl) - 1), 2):
                    circuits_string.append('CX %s %s' % (spl[ii+1], spl[ii+2]))
            elif gate_name == 'CZ':
                for ii in range(0, (len(spl) - 1), 2):
                    circuits_string.append('CZ %s %s' % (spl[ii+1], spl[ii+2]))
            # one qubit gate
            elif gate_name == 'H' or gate_name == 'S' or gate_name == 'SQRT_X' or gate_name == 'X':
                for ii in range(1, len(spl)):
                    circuits_string.append('%s %s' % (gate_name, spl[ii]))
            else:
                # does not handle other clifford gates and non-clifford gate
                raise NotImplementedError()
        return circuits_string   

    def get_propagated_errors (self):
        """ Going through stim enconding circuit and apply error and see the propagated errors.
        Very inefficient  but only needs to be done once.
    
        Returns:
            Splitted Stim circuit.    
        """
        
        ## Split the circuit to enable insertion of error
        circuits_string = self.split_stim_circuit()
        
        ## Propagate errors 
        propagated_errors = [] 
        for ii, circ in enumerate(circuits_string):

            # For one qubit gate
            if len(circ.split(' ')) == 2:
                ## Apply error before the gate is applied
                for error_type in ['X', 'Y', 'Z']:  
                    propagated_error = self.propagate_error(int(circ.split(' ')[1]), error_type, circuits_string[ii:])
                    propagated_errors.append(propagated_error)

            else: ## for cnot need to put error in both
                for error_type in self.two_qubit_errors: 
                    propagated_error = self.propagate_error(circ.split(' ')[1:], error_type, circuits_string[ii:])
                    propagated_errors.append(propagated_error)

        ## Convert to PauliString
        propagated_errors_pauli = PauliString([str(string)[1:] + ("I" * self.num_ancillas) for string in propagated_errors])

        ## For distance 5 code, need to combine error in two gates
        ## Very slow
        if self.distance == 5:
            propagated_errors_pauli = propagated_errors_pauli.multiply_each()

        return propagated_errors_pauli    
    
    def propagate_error(self, error_location, error_type, circuits_string):
        """ Propagate error from given circuit.

        Args:
            error_location: location of the error or a list for multi-qubit error.
            error_type: X, Y, or Z error or a list for multi-qubit error.
            circuits_string: string of the circuit.

        Returns:
            Propagated error.
        """
        error_to_propagate = stim.PauliString(self.n_qubits_physical_encoding)
        ## Multi qubit error
        if isinstance(error_location, list):
            error_to_propagate[int(error_location[0])] = error_type[0]
            error_to_propagate[int(error_location[1])] = error_type[1]
        ## Single qubit error
        else:
            error_to_propagate[error_location] = error_type

        circuit = stim.Circuit('\n'.join(circuits_string))
        propagated_error = error_to_propagate.after(circuit)
        return propagated_error

    def action_matrix(self) -> chex.Array:
        """ Generate the action matrix.

        Returns:
            Action matrix for each gate.
        """
        self.action_matrix = []
        self.action_string = []
        self.action_string_stim = []
        self.action_string_stim_circ = []
        self.action_errors = []  ## errors to add when applying action

        self.sign_matrix = []
        self.sign_mask = []

        for gate in self.gates:
            ## One qubit gate
            if len(signature(gate).parameters) == 1:
                for n_qubit in range(self.n_qubits_physical): 
                    
                    ## Ignore gate in the data qubits
                    if not self.gates_between_data and n_qubit < self.n_qubits_physical_encoding:
                        continue            
                        
                    self.action_matrix.append(gate(n_qubit)[0])
                    self.action_string.append('%s-%d' % (gate.__name__, n_qubit))
                    self.action_string_stim.append('.%s(%d)' % (gate.__name__.lower(), n_qubit))
                    self.action_string_stim_circ.append('.append("%s", [%d])' % (gate.__name__.lower(), n_qubit))
                    
                    self.sign_matrix.append(gate(n_qubit)[1])                    
                    self.sign_mask.append(0)

                    ## Add error 
                    action_error_str = []
                    for error_type in ['X', 'Y', 'Z']:
                        error_str = list("I" * self.n_qubits_physical)
                        error_str[n_qubit] = error_type
                        
                        action_error_str.append(''.join(error_str))

                    ### Add extra errors of IIII...I to make the shape of action error the same (not dynamics)
                    ### There are 15 errors following the two qubit gates
                    for ii in range(12):
                        error_str = list("I" * self.n_qubits_physical)
                        action_error_str.append(''.join(error_str))

            ## Two qubit gates
            elif len(signature(gate).parameters) == 2:

                for edge in self.graph:
                    ## Ignore some edge if it is not connected to ancilla
                    if not self.gates_between_data and (edge[0] < self.n_qubits_physical_encoding and edge[1] < self.n_qubits_physical_encoding) :     
                        continue

                    ## Two qubit gates between ancilla
                    if not self.gates_between_ancilla and (edge[0] >= self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding):
                        continue

                    ## Ignore some edge If ancilla is only the target of two qubit gate
                    if self.ancilla_target_only and edge[1] < self.n_qubits_physical_encoding and edge[0] >= self.n_qubits_physical_encoding:
                        continue

                    ## Ignore some edge If ancilla is only the target of two qubit gate
                    if self.ancilla_control_only and edge[0] < self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding:
                        continue

                    ### Grouping ancillas into two
                    if self.group_ancillas and edge[0] >= self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding:
                        if ((edge[0] - self.n_qubits_physical_encoding) // 2) != ((edge[1] - self.n_qubits_physical_encoding) // 2): 
                            continue

                    ### Grouping ancillas avoid gate to the flag ancilla
                    if self.group_ancillas and    \
                        ((edge[0] < self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding and ((edge[1] - self.n_qubits_physical_encoding) % 2) == 1) or  \
                        (edge[0] >= self.n_qubits_physical_encoding and edge[1] < self.n_qubits_physical_encoding and ((edge[0] - self.n_qubits_physical_encoding) % 2) == 1)):
                        continue

                    if self.group_ancillas and    \
                        (edge[0] >= self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding and edge[1] < edge[0]):
                        continue

                    if self.group_ancillas and    \
                        (edge[0] >= self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding and edge[1] > edge[0] and gate.__name__ == 'cz'):
                        continue

                    ## Avoid unnecessary cz since it is symmetric
                    if gate.__name__ == 'cz' and 'cz-%d-%d' % (edge[1], edge[0]) in self.action_string:
                        continue
                    self.action_matrix.append(gate(edge[0], edge[1])[0])
                    self.action_string.append('%s-%d-%d' % (gate.__name__, edge[0], edge[1]))
                    self.action_string_stim.append('.%s(%d, %d)' % (gate.__name__.lower(), edge[0], edge[1]))
                    self.action_string_stim_circ.append('.append("%s", [%d, %d])' % (gate.__name__.lower(), edge[0], edge[1]))
                    
                    self.sign_matrix.append(gate(edge[0], edge[1])[1])
                    self.sign_mask.append(1)

                    ## Add error 
                    action_error_str = []
                    for error_type in self.two_qubit_errors:
                        error_str = list("I" * self.n_qubits_physical)
                        error_str[edge[0]] = error_type[0]
                        error_str[edge[1]] = error_type[1]
                        action_error_str.append(''.join(error_str))

                    self.action_errors.append(PauliString(action_error_str).to_numpy())

        self.action_matrix = jnp.array(self.action_matrix)
        self.action_errors = jnp.array(self.action_errors)
        self.sign_matrix = jnp.array(self.sign_matrix, dtype=jnp.uint8)
        self.sign_mask = jnp.array(self.sign_mask, dtype=jnp.uint8)

        return self.action_matrix
  
    def measure_flag(self, propagated_error):
        """ Measure the flag of the list of propagated error to calculate the flag reward f_t.
        
        Args:
            propagated_error: The list of error propagated to measure the flag.

        Returns:
            The flag value for each propagated error.
        """
        
        if self.mul_errors_with_generators or self.mul_errors_with_S:
            propagated_error = self.multiply_propagated_error(propagated_error)

        ## Get weight
        propagated_error_weight = propagated_error.get_weight(index=jnp.arange(0, self.n_qubits_physical_encoding), ignore_x=self.ignore_x_errors, ignore_y=self.ignore_y_errors, ignore_z=self.ignore_z_errors)

        ancilla_weight_zero = propagated_error.get_weight(index=self.zero_ancilla_position, ignore_z=True) ## check if ancilla is X or Y in the zero qubit
        ancilla_weight_plus = propagated_error.get_weight(index=self.plus_ancilla_position, ignore_x=True) ## check if ancilla is Z or Y in the plus qubit

        ancilla_weight = ancilla_weight_zero + ancilla_weight_plus
        correctible = (propagated_error_weight <= (self.distance - 1) // 2) 
        
        uncorrectible_flipped =  (propagated_error_weight > (self.distance - 1) // 2)  & (ancilla_weight > 0)
        result = jnp.where(correctible |  uncorrectible_flipped, 1, 0)

        return result

    def multiply_propagated_error(self, propagated_error):
        ''' Update propagated error by multiplying with generators or S.

        Args:
            propagated_error: The list of error propagated to multiply.

        Return:
            Updated propagated error after multiplied with the generators or S.
        '''    
        ## Create pauli string for generators   
        generators_pauli = PauliString()
        if self.mul_errors_with_generators:

            ## Get generator of current tableau
            generators = self.encoding_generators.to_numpy()

            ## Add IIIII...I to multiply
            generators = jnp.vstack((jnp.zeros((1, generators.shape[1])), generators))

            ## set ancilla to I 
            generators = generators.at[:, self.n_qubits_physical - self.num_ancillas: self.n_qubits_physical].set(0)
            generators = generators.at[:, self.n_qubits_physical * 2 - self.num_ancillas:].set(0)

            generators_pauli.from_numpy(generators)

        elif self.mul_errors_with_S:
            ## Get generator of current tableau
            generators = self.encoding_group.to_numpy()

            
            ## set ancilla to I            
            generators = generators.at[:, self.n_qubits_physical - self.num_ancillas: self.n_qubits_physical].set(0)
            generators = generators.at[:, self.n_qubits_physical * 2 - self.num_ancillas:].set(0)

            generators_pauli.from_numpy(generators)

        propagated_error = propagated_error.multiply_and_update(generators_pauli, num_ancillas=self.num_ancillas, ignore_x=self.ignore_x_errors, ignore_y=self.ignore_y_errors, ignore_z=self.ignore_z_errors)

        return propagated_error

    def count_flagged_errors(self, propagated_error):
        """Count flagged errors that will return f_t.
        
        Args:
            propagated_error: The list of error propagated to count flagged errors.
        
        Returns:
            f_t value between 0 to 1.
        """
        ## Update propagated errors from a given action

        propagated_errors = PauliString().from_numpy(propagated_error)

        return jnp.mean(self.measure_flag(propagated_errors))
    
    def flagged_errors(self, propagated_error):
        """ Get flagged errors for debugging purposes.

        Args:
            propagated_error: The list of error propagated to count flagged errors.

        Returns:
            The flagged errors

        """
        propagated_errors = PauliString().from_numpy(propagated_error)
        return self.measure_flag(propagated_errors)   

    def update_propagated_errors(self, propagated_error, action, number_of_errors):
        """ Update propagated errors from a given action.

        Args:
            propagated_error: Propagated errors to be updated
            action (int): The action that will be applied
            number_of_errors (int): The current number of errors.

        Returns:
            Updated given propagated errors.
        """

        propagated_errors = PauliString().from_numpy(propagated_error)

        action_errors = PauliString().from_numpy(self.action_errors[action])
        propagated_errors = propagated_errors.update_pauli(action_errors, number_of_errors)

        ## Update propagated error
        propagated_errors = propagated_errors.after(self.action_matrix[action])
        

        return propagated_errors

    def jaccard(self, vec1, vec2):
        """ Compute jaccard distance of the tableau for the reward.
        
        Args:
            vec1: Vector 1 input.
            vec2: Vector 2 input.
        
        Returns:
            Jaccard distance between vec1 and vec2.
        """
        intersection = jnp.logical_and(vec1, vec2)
        union = jnp.logical_or(vec1, vec2)
        return jnp.sum(intersection) / jnp.sum(union)

    def get_distance(self,tableaus, signs):        
        ''' Evaluate the distance with the target tableau for the reward.
        
        Args:
            tableaus: The tableau of the current quantum circuit.
            signs: The sign of stabilizers of the current quantum circuit.
        
        Returns:
            Distance reward d_t.
        '''
        canon_check_matrix, canon_sign = self.canonical_stabilizers(self.get_observation_without_ancilla(tableaus), signs[self.n_qubits_physical:self.n_qubits_physical+self.n_qubits_physical_encoding]  * 2) 
        
        current_check_matrix = jnp.append(canon_check_matrix.flatten(), canon_sign // 2)

        code_distance = self.jaccard(current_check_matrix, self.target_check_matrix) 

        return code_distance
    
    def update_signs(self, previous_tableau, current_tableau, sign, action):
        """ Updating the sign of the tableau.

        Args:
            previous_tableau: The tableau from the previous action.
            current_tableau: The tableau after applying the action.
            sign: The sign from the previous action.
            action: The action applied
        
        Return: 
            Updated sign after applying action
        """
        theta = self.sign_mask[action] 
        
        # The structure is:
        # theta * diag(curr_tabl @ sign_matrix[action,0] @ curr_tabl.T) * diag(curr_tabl @ sign_matrix[action,1] @ curr_tabl.T) + 
        # + diag(prev_tabl @ sign_matrix[action,0] @ prev_tabl.T) * diag(prev_tabl @ sign_matrix[action,1] @ prev_tabl.T)
        
        # theta is zero for one-qubit gates, which means they are updated using the previous tableau
        # Previous and current is needed for cx because sign flips both coming and going to YY
        # sign_matrix[action,0] and sign_matrix[action,1] are needed to check that Y(c) and Y(t) are there
        # for single-qubit gates I'm just redundantly doing Y(i)*Y(i), which creates overhead
        
        signs_update = (theta * (jnp.diagonal(current_tableau @ self.sign_matrix[action,0] @ jnp.transpose(current_tableau))) 
                         * jnp.diagonal(current_tableau @ self.sign_matrix[action,1] @ jnp.transpose(current_tableau)) 
                         + jnp.diagonal(previous_tableau @ self.sign_matrix[action,0] @ jnp.transpose(previous_tableau))
                         * jnp.diagonal(previous_tableau @ self.sign_matrix[action,1] @ jnp.transpose(previous_tableau))).astype(jnp.uint8)
        # This part if pretty trivial
        sign += signs_update
        sign %= 2  

        return sign
    
    def swap_matrix(self, num_qubits, pos1, pos2):
        """ Create swap matrix that swap row of position 1 (pos1) and position 2 (pos2) for gaussian elimination.

        Args:
            num_qubits (int): Number of qubits
            pos1(int): Position 1
            pos2(int): Position 2

        Returns:
            A matrix that applied to the tableau swap row of pos1 and pos2.
        """

        swap = jnp.eye(num_qubits)
        swap = swap.at[pos1,pos1].set(0)
        swap = swap.at[pos2,pos2].set(0)
        swap = swap.at[pos1,pos2].set(1)
        swap = swap.at[pos2,pos1].set(1)
        return swap

    def ipow(self, check_matrix, other_check_matrix):
        """ Phase indicator for the product of two Pauli strings.

        Args:
            check_matrix: the check matrix.
            other_check_matrix: the other check matrix.
        
        Returns:
            ipow (int): the phase indicator (power of i).
        """
        num_qubits = check_matrix.shape[1] // 2
        g1x = check_matrix[:,:num_qubits]
        g1z = check_matrix[:,num_qubits:]
        g2x = other_check_matrix[:,:num_qubits]
        g2z = other_check_matrix[:,num_qubits:]
        
        gx = g1x + g2x
        gz = g1z + g2z
        
        multiply = (g1z * g2x) - (g1x *g2z) + 2 * ((gx // 2) * gz + gx * (gz // 2))
        return jnp.sum(multiply,1) % 4

    def canonical_stabilizers(self, check_matrix, sign):
        ''' Gaussian elimination to get canonical stabilizers

        Args: 
            check_matrix: Check matrix of the current tableau.
            sign: sign of the current tableau

        Returns:
            Canonical tableau and its sign
        
        '''
        num_qubits = check_matrix.shape[0]
        min_pivot = 0
        ## Loop through rows
        for q in range(num_qubits):
            ## Loop through columns x z x z ...
            for b in range(2):

                #### 1. Find pivot
                ## Pivot row
                pivot = min_pivot

                ## Find position of 1 in the current column larger than pivot
                ## To vectorize, the trick is to use mask to get value larger than the pivot
                ## and the argmax should return the position of the first 1 otherwise it is 0, which means that we can skip this column
                check_column = check_matrix[:, q+b*num_qubits].astype(jnp.uint8)
                mask = jnp.array(range(num_qubits))
                mask = jnp.where(mask >= pivot, 1, 0).astype(jnp.uint8)
                ## Get pivot value
                max_value = jnp.max(check_column * mask, initial=0)
                pivot = jnp.argmax(check_column * mask) 
                
                #### 2. Remove ones in the column other than pivot
                ## Create the matrix to eliminate ones
                elim = jnp.eye(num_qubits)
                elim = elim.at[:,pivot].set(check_column)
                condition = max_value
                
                check_matrix = (condition * ((elim @ check_matrix) % 2)) + ((1- condition) * check_matrix)
                
                #### 3.  Update sign
                check_column = check_column.at[pivot].set(0)
                repeat_pivot = jnp.repeat(pivot, num_qubits)
                sign = sign + (condition * check_column * ((sign[repeat_pivot] + self.ipow(check_matrix, check_matrix[repeat_pivot])) )) 
                sign = sign % 4

                #### 4. Swap the pivot to the min pivot
                condition = (min_pivot != pivot).astype(jnp.uint8) * max_value
                swap = self.swap_matrix(num_qubits, min_pivot, pivot)
                check_matrix = (condition * swap @ check_matrix) + (1-condition) * check_matrix
                sign = (condition * swap @ sign) + (1-condition) * sign
                
                min_pivot += 1 * max_value


        return check_matrix.astype(jnp.uint8), sign.astype(jnp.uint8)
 
    def check_ancilla_product_state(self, tableaus):
        """ Check that the state is a product state with ancilla by checking that the last column and row of the ancilla in the canonical tableau is either I or Z only.
        This counts the product state reward p_t.

        Args:
            tableaus: Input tableaus to check for the ancilla product state.

        Returns:
            The product state reward p_t
        """
        
        canon_check_matrix, canon_sign = self.canonical_stabilizers(self.get_observation(tableaus), jnp.zeros(self.n_qubits_physical)  * 2) 
        ## Get check matrix of the last columns
        check_mat_ancilla_x = canon_check_matrix[:, self.n_qubits_physical - self.num_ancillas: self.n_qubits_physical].astype(jnp.uint8)
        check_mat_ancilla_z = canon_check_matrix[:, 2 * self.n_qubits_physical - self.num_ancillas:].astype(jnp.uint8)
        check_mat_ancilla = jnp.block([check_mat_ancilla_x, check_mat_ancilla_z]).astype(jnp.uint8)

        ## Get the pauli strings
        ancilla_paulis = PauliString().from_numpy(check_mat_ancilla)
        column_ancillas = jnp.sum(ancilla_paulis.get_inverse_weight(index=self.zero_ancilla_position - self.n_qubits_physical_encoding, include_z = True))  + jnp.sum(ancilla_paulis.get_inverse_weight(index=self.plus_ancilla_position - self.n_qubits_physical_encoding, include_x = True))  
        column_ancillas = column_ancillas / (self.n_qubits_physical * self.num_ancillas)
        # return column_ancillas

        # # ## Get check matrix of the last rows
        check_mat_ancilla_x = canon_check_matrix[self.n_qubits_physical - self.num_ancillas:, :self.n_qubits_physical - self.num_ancillas].astype(jnp.uint8)
        check_mat_ancilla_z = canon_check_matrix[self.n_qubits_physical - self.num_ancillas:, self.n_qubits_physical: 2 * self.n_qubits_physical - self.num_ancillas].astype(jnp.uint8)
        check_mat_ancilla = jnp.block([check_mat_ancilla_x, check_mat_ancilla_z]).astype(jnp.uint8)
        # ## Get the pauli strings
        ancilla_paulis = PauliString().from_numpy(check_mat_ancilla)
        row_ancillas = jnp.sum(ancilla_paulis.get_inverse_weight(include_z = True)) /  (self.n_qubits_physical_encoding * self.num_ancillas)
        
        ## Count the I and Z
        return (column_ancillas + row_ancillas) / 2.
      
    def step_env(self, key: chex.PRNGKey, state: EnvState, action: int, params=None) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment.

        Args:
            key: Random key for Jax.
            state: The current state.
            action: The action to be applied.
            params: Parameters.

        Returns: 
            new observation, new state, reward, done, information.
        """
    
        # Update state
        new_state = jnp.matmul(state.tableau, self.actions[action]) % 2
        new_sign = self.update_signs(state.tableau, new_state, state.signs, action)

        # Update propagated errors
        new_propagated_error = self.update_propagated_errors(state.propagated_error, action, state.number_of_errors).to_numpy()
        
        ## Add 15 new errors for the newly applied gates
        new_number_of_errors = state.number_of_errors + 15

        current_flagged_errors = self.count_flagged_errors(new_propagated_error)
        current_product_ancilla = self.check_ancilla_product_state(new_state)
        current_distance = self.get_distance(new_state, new_sign)

        reward = self.weight_ancillas * (current_product_ancilla - state.previous_product_ancilla) +  \
                        self.weight_flag * (current_flagged_errors - state.previous_flagged_errors) +     \
                        self.weight_distance * (current_distance - state.previous_distance)
        

        state = EnvState(new_state.astype(jnp.uint8),  new_sign.astype(jnp.uint8), new_propagated_error.astype(jnp.uint8), current_flagged_errors,  current_product_ancilla, current_distance, new_number_of_errors, state.time + 1)

        # Evaluate termination conditions
        done = self.is_terminal(state)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,           
            {"discount": self.discount(state, params)},
        )

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment.

        Args:
            key: Random key for Jax.
            params: Parameters.

        Returns: 
            observation, state.
        """
        ## Tableau simulators 
        tableaus = TableauSimulator(self.n_qubits_physical,initial_tableau=self.initial_tableau_with_ancillas.current_tableau[0], initial_sign=self.initial_tableau_with_ancillas.current_signs[0])

        ## Hardcode qubits into +
        for ii in self.plus_ancilla_position:
            tableaus.h(ii)

        propagated_errors = self.initial_propagated_errors.copy() 
        previous_flagged_errors = jnp.mean(self.measure_flag(propagated_errors))
        previous_distance = self.get_distance(tableaus.current_tableau[0], tableaus.current_signs[0])

        previous_product_ancilla = self.check_ancilla_product_state(tableaus.current_tableau[0])

        state = EnvState(
            tableau = tableaus.current_tableau[0],
            signs = tableaus.current_signs[0],
            propagated_error = propagated_errors.to_numpy(),
            previous_flagged_errors=previous_flagged_errors,
            previous_product_ancilla=previous_product_ancilla,
            previous_distance=previous_distance,
            number_of_errors = self.number_of_initial_errors,
            time = 0
        )

        return self.get_obs(state), state
    
    def is_terminal(self, state: EnvState, params=None) -> bool:
        """Check whether state is terminal.
        
        Args:
            state: The state.
            params: The parameters.
        
        Return:
            True if the distance is more than threshold or the time is more than max_steps.
        """
        check = (state.previous_flagged_errors + state.previous_distance + state.previous_product_ancilla) / 3.
        
        done_encoding = check > self.threshold
        
        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps
        
        done = jnp.logical_or(done_encoding, done_steps)
        return done
    
    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state.

        Args:
            state: The state.
            params: The parameters.
        
        Returns:
            Observations by appending the tableau and the sign
        """
        return self.get_observation(state.tableau).flatten()
    
    def copy(self):
        """ Copy environment. """
        return VerificationCircuitSynthesisEnv(
            self.encoding_circuit,
            self.num_ancillas,    
            self.distance,      
            self.gates,
            self.graph,   
            self.max_steps,
            self.threshold,                
            self.mul_errors_with_generators,
            self.mul_errors_with_S,
            self.ignore_x_errors,
            self.ignore_y_errors,
            self.ignore_z_errors,
            self.weight_distance,
            self.weight_flag,
            self.weight_ancillas,
            self.ancilla_target_only,
            self.ancilla_control_only,
            self.gates_between_ancilla,
            self.gates_between_data,
            self.group_ancillas,
            self.plus_ancilla_position
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "VerificationCircuitSynthesisEnv"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.actions.shape[0]

    def action_space(self, params: Optional[EnvParams] = EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape, dtype=jnp.uint8)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "tableau": spaces.Box(0, 1, self.obs_shape, jnp.uint8),
                "time": spaces.Discrete(self.max_steps),
            }
        )

    @property
    def default_params(self) -> EnvParams:
        """ Default environment parameters. """
        return EnvParams()   