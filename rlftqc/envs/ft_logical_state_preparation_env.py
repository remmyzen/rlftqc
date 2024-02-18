from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
from flax import struct
import chex
from inspect import signature
from typing import Tuple, Optional
import itertools
from rlftqc.simulators import PauliString, TableauSimulator, CliffordGates
from rlftqc.utils import is_css_code


@struct.dataclass
class EnvState:
    """ This class will contain the state of the environment.
    """
    tableau: jnp.array
    sign: jnp.array
    propagated_error: jnp.array
    previous_flagged_errors: float
    previous_distance: float
    previous_product_ancilla: float
    number_of_errors: int
    time: int
    
@struct.dataclass
class EnvParams:
    max_steps: int = 60

class FTLogicalStatePreparationEnv(environment.Environment):
    """Environment for the integrated fault-tolerant logical state preparation task."""

    def __init__(self,
            target,
            num_ancillas = 1, 
            distance = 3,
            gates=None, 
            graph=None,  
            max_steps = 50,
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
            group_ancillas = False,  
            cz_ancilla_only = False,
            plus_ancilla_position = [],
            distance_metric = 'jaccard'
        ):
        """ Initialize a integrated fault-tolerant logical state preparation environment.
        
        Args:
            target (list(str)): List of stabilizers of the target state as a string.
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
            group_ancillas (boolean, optional): If set to True, this will group ancilla into two. Useful to replicate the protocol in Chamberland and Chao original paper. For example: If there are 4 flag qubits, there will be no two-qubit gates between flag qubits 1,2 and 3,4. 
            plus_ancilla_position (list(int), optional): Initialize flag qubits given in the list as plus state and will measure in the X basis.
                This is useful for non-CSS codes.
            cz_ancilla_only, boolean, optional): If true, then CZ only applied in the ancilla. (Default: False)
            distance_metric (str, optional): Distance metric to use for the complementary distance reward.
                Currently only support 'hamming' or 'jaccard' (default).
        """
        super().__init__()

        ### Process target stabilizers
        self.target = target
        self.target_sign = []
        self.n_qubits_physical_encoding = len(self.target)
        target_wo_sign = []
        for stabs in self.target:
            ## Process sign
            if '+' in stabs[0]:
                self.target_sign.append(0)
                stab_wo_sign = stabs[1:].upper()
            elif '-' in stabs[0]:
                self.target_sign.append(1)
                stab_wo_sign = stabs[1:].upper()
            else:
                self.target_sign.append(0)
                stab_wo_sign = stabs.upper()

            if len(stab_wo_sign) != self.n_qubits_physical_encoding:
                raise ValueError("The length of stabilizer",stabs, "is not correct.")

            if not all(char in ['I', 'X', 'Y', 'Z'] for char in stab_wo_sign):
                raise ValueError("Stabilizer contains Pauli string other than I, X, Y, or Z.")

            target_wo_sign.append(stab_wo_sign)
        self.target_tableau = PauliString(target_wo_sign)

        self.is_css = is_css_code(self.target)

        self.num_ancillas = num_ancillas
        self.n_qubits_physical = self.n_qubits_physical_encoding + self.num_ancillas

        ## Process gates
        self.gates = gates
        if self.gates is None:
            clifford_gates = CliffordGates(self.n_qubits_physical)
             ## Initialize with the standard gate set + CZ minus S for non-CSS
            if self.is_css:
                print("Code is CSS, but gate set is not specified using H, S, and CX as the default gate set.")
                self.gates = [clifford_gates.h, clifford_gates.s, clifford_gates.cx] 
            else:               
                print("Code is non-CSS, but gate set is not specified using H, S, CX, and CZ as the default gate set.")
                self.gates = [clifford_gates.h, clifford_gates.s, clifford_gates.cz, clifford_gates.cx] 


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
        self.distance_metric = distance_metric

        self.mul_errors_with_generators = mul_errors_with_generators ## multiply error with generators might reduce number of errors but make the process slower
        self.mul_errors_with_S = mul_errors_with_S ## multiply error with S

        if not self.is_css and not self.mul_errors_with_S:
            print("Code is non-CSS, multiply errors with S is set to True. It might need a big memory for bigger codes.")
            self.mul_errors_with_S = True
        
        if self.mul_errors_with_S and self.mul_errors_with_generators:
            self.mul_errors_with_generators = False

        self.weight_distance = weight_distance
        if self.weight_distance is None:
            self.weight_distance = self.n_qubits_physical_encoding 

        self.weight_flag = weight_flag
        if self.weight_flag is None:
            self.weight_flag = self.n_qubits_physical_encoding // 2

        self.weight_ancillas = weight_ancillas
        if self.weight_ancillas is None:
            self.weight_ancillas = 1.

        self.ignore_x_errors = ignore_x_errors
        self.ignore_y_errors = ignore_y_errors
        self.ignore_z_errors = ignore_z_errors    
        
        if self.is_css and not self.ignore_x_errors and not self.ignore_y_errors and not self.ignore_z_errors:
            print("WARNING! Code is CSS but no errors are ignored. You might want to set to ignore some error depending on the logical state that you are preparing. For example: For zero-logical, then Z errors can be ignored and add ignore_z_errors = True as an argument.")
   
        self.ancilla_target_only = ancilla_target_only
        self.ancilla_control_only = ancilla_control_only
        self.gates_between_ancilla = gates_between_ancilla
        self.cz_ancilla_only = cz_ancilla_only

        self.group_ancillas = group_ancillas
        self.plus_ancilla_position = jnp.array(plus_ancilla_position, dtype=jnp.int32)
        self.zero_ancilla_position = jnp.delete(jnp.arange(self.n_qubits_physical_encoding, self.n_qubits_physical_encoding+self.num_ancillas),  self.plus_ancilla_position  - self.n_qubits_physical_encoding)
        
                
        ### Process encoding tableau
        self.initial_tableau_with_ancillas = TableauSimulator(self.n_qubits_physical)
        
        self.target_tableau_with_ancilla = PauliString([string + ("I" * self.num_ancillas) for string in target_wo_sign])
        
        self.target_check_matrix_unflatten = jnp.array(self.target_tableau.to_numpy())

        canon_target_check_matrix, canon_target_sign = self.canonical_stabilizers(self.target_check_matrix_unflatten, jnp.array(self.target_sign) * 2)
        self.target_check_matrix = jnp.append(canon_target_check_matrix.flatten(), canon_target_sign // 2)

        ## Only generate group if need to mul error with S
        if self.mul_errors_with_S:
            self.target_tableau_with_ancilla_group = self.target_tableau_with_ancilla.generate_stabilizer_group()
        
        ## Get propagated error from the encoding circuit
        # Generate all two qubit errors except II...I
        self.two_qubit_errors = list(itertools.product('IXYZ', repeat=2))[1:]  
        self.initial_propagated_errors = PauliString(self.n_qubits_physical, 15 * self.max_steps)

        self.actions = self.action_matrix()

        self.obs_shape = self.get_observation(self.initial_tableau_with_ancillas.current_tableau[0]).flatten().shape[0] + self.n_qubits_physical


    def get_observation(self, tableau):
        """ Extract the check matrix for the observation of the RL agent.
        Args:
            tableau: check matrix of the tableau.
        Return:
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

    def get_generators_from_tableau(self, tableau):
        """ Extract generators check matrix from a given tableau.
        
        Args:
            tableau: tableau to be extracted.

        Return:
            Generators of the tableau.
        """
        check_mat = tableau[self.n_qubits_physical:self.n_qubits_physical + self.n_qubits_physical_encoding].astype(jnp.uint8)

        return check_mat

   
    def action_matrix(self) -> chex.Array:
        '''
        Generate the possible actions
        '''
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

                    ## If there is a forced hadamard in the ancilla ignore hadamard on ancilla
                    if len(self.plus_ancilla_position) > 0 and n_qubit >= self.n_qubits_physical_encoding and n_qubit in self.plus_ancilla_position:
                        continue
                    
                    self.action_matrix.append(gate(n_qubit)[0])
                    self.sign_matrix.append(gate(n_qubit)[1])                    
                    self.sign_mask.append(0)

                    self.action_string.append('%s-%d' % (gate.__name__, n_qubit))
                    self.action_string_stim.append('.%s(%d)' % (gate.__name__.lower(), n_qubit))
                    self.action_string_stim_circ.append('.append("%s", [%d])' % (gate.__name__.lower(), n_qubit))
                    
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
                    ## Cnot between ancilla
                    if not self.gates_between_ancilla and (edge[0] >= self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding):
                        continue

                    ## Ignore some edge If ancilla is only the target of two qubit gate
                    if self.ancilla_target_only and edge[1] < self.n_qubits_physical_encoding and edge[0] >= self.n_qubits_physical_encoding:
                        continue

                    ## Ignore some edge If ancilla is only the control of two qubit gate
                    if self.ancilla_control_only and edge[0] < self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding:
                        continue

                    ### Grouping ancillas into two
                    if self.group_ancillas and edge[0] >= self.n_qubits_physical_encoding and edge[1] >= self.n_qubits_physical_encoding:
                        if ((edge[0] - self.n_qubits_physical_encoding) // 2) != ((edge[1] - self.n_qubits_physical_encoding) // 2): 
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


                    ## CZ to ancilla only
                    if gate.__name__ == 'cz' and self.cz_ancilla_only and edge[0] < self.n_qubits_physical_encoding and edge[1] < self.n_qubits_physical_encoding:
                        continue

                    ## Avoid unnecessary cz
                    if gate.__name__ == 'cz' and 'cz-%d-%d' % (edge[1], edge[0]) in self.action_string:
                        continue
                    

                    self.action_matrix.append(gate(edge[0], edge[1])[0])     
                    self.sign_matrix.append(gate(edge[0], edge[1])[1])
                    self.sign_mask.append(1)

                    self.action_string.append('%s-%d-%d' % (gate.__name__, edge[0], edge[1]))
                    self.action_string_stim.append('.%s(%d, %d)' % (gate.__name__.lower(), edge[0], edge[1]))
                    self.action_string_stim_circ.append('.append("%s", [%d, %d])' % (gate.__name__.lower(), edge[0], edge[1]))
                    
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
 
    def measure_flag(self, tableaus, propagated_error):
        """ Measure the flag of the list of propagated error to calculate the flag reward f_t.
        
        Args:
            tableaus: Tableaus to multiply errors to reduce the weight.
            propagated_error: The list of error propagated to measure the flag.

        Returns:
            The flag value for each propagated error.
        """
        
        if self.mul_errors_with_generators or self.mul_errors_with_S:
            propagated_error = self.multiply_propagated_error(tableaus, propagated_error)

        ## Get weight
        propagated_error_weight = propagated_error.get_weight(index=jnp.arange(0, self.n_qubits_physical_encoding), ignore_x=self.ignore_x_errors, ignore_y=self.ignore_y_errors, ignore_z=self.ignore_z_errors)

        ancilla_weight_zero = propagated_error.get_weight(index=self.zero_ancilla_position, ignore_z=True) ## check if ancilla is X or Y in the zero qubit
        ancilla_weight_plus = propagated_error.get_weight(index=self.plus_ancilla_position, ignore_x=True) ## check if ancilla is Z or Y in the plus qubit

        ancilla_weight = ancilla_weight_zero + ancilla_weight_plus
        correctible = (propagated_error_weight <= (self.distance - 1) // 2) 
        
        uncorrectible_flipped =  (propagated_error_weight > (self.distance - 1) // 2)  & (ancilla_weight > 0)
        result = jnp.where(correctible |  uncorrectible_flipped, 1, 0)

        return result

    def multiply_propagated_error(self, tableau, propagated_error):
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
            generators = self.get_generators_from_tableau(tableau)

            ## Add IIIII...I to multiply
            generators = jnp.vstack((jnp.zeros((1, generators.shape[1])), generators))

            ## set ancilla to I 
            generators = generators.at[:, self.n_qubits_physical - self.num_ancillas: self.n_qubits_physical].set(0)
            generators = generators.at[:, self.n_qubits_physical * 2 - self.num_ancillas:].set(0)

            generators_pauli.from_numpy(generators)

        elif self.mul_errors_with_S:
            ## Get generator of current tableau
            generators = self.get_generators_from_tableau(tableau)

            
            ## set ancilla to I            
            generators = generators.at[:, self.n_qubits_physical - self.num_ancillas: self.n_qubits_physical].set(0)
            generators = generators.at[:, self.n_qubits_physical * 2 - self.num_ancillas:].set(0)

            generators_pauli.from_numpy(generators)
            generators_pauli = generators_pauli.generate_stabilizer_group()

        propagated_error = propagated_error.multiply_and_update(generators_pauli, num_ancillas=self.num_ancillas, ignore_x=self.ignore_x_errors, ignore_y=self.ignore_y_errors, ignore_z=self.ignore_z_errors)

        return propagated_error

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

    def count_flagged_errors(self, tableaus, propagated_error, number_of_errors):
        """ Count flagged errors that will return f_t.
        
        Args:
            tableaus: The current tableau
            propagated_error: The list of error propagated to count flagged errors.
            number_of_errors: Current number of errors.

        Returns:
            f_t value between 0 to 1.
        """
        ## Update propagated errors from a given action
        propagated_errors = PauliString().from_numpy(propagated_error)
        measured_flag = self.measure_flag(tableaus, propagated_errors)

        return (jnp.sum(measured_flag) - (jnp.shape(measured_flag)[0] - number_of_errors)) /  number_of_errors
 
    def check_ancilla_product_state(self, tableaus, signs):
        """ Check that the state is a product state with ancilla by checking that the last column and row of the ancilla in the canonical tableau is either I or Z only.
        This counts the product state reward p_t.

        Args:
            tableaus: Input tableaus to check for the ancilla product state.

        Returns:
            The product state reward p_t
        """
        
        canon_check_matrix, canon_sign = self.canonical_stabilizers(self.get_observation(tableaus), signs[self.n_qubits_physical:]  * 2) 
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
      
    def hamming(self, vec1, vec2):
        """ Compute hamming distance of the tableau for the reward.
        
        Args:
            vec1: Vector 1 input.
            vec2: Vector 2 input.
        
        Returns:
            Hamming distance between vec1 and vec2.
        """
        return (jnp.shape(vec1)[0] - jnp.count_nonzero(vec1 != vec2)) / jnp.shape(vec1)[0]

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

    def get_distance(self, tableaus, signs):
        ''' Evaluate the distance with the target tableau for the reward.
        
        Args:
            tableaus: The tableau of the current quantum circuit.
            signs: The sign of stabilizers of the current quantum circuit.
        
        Returns:
            Distance.
        '''
        canon_check_matrix, canon_sign = self.canonical_stabilizers(self.get_observation_without_ancilla(tableaus), signs[self.n_qubits_physical:self.n_qubits_physical+self.n_qubits_physical_encoding]  * 2) #
        current_check_matrix = jnp.append(canon_check_matrix.flatten(), canon_sign // 2)

        if self.distance_metric == 'jaccard':
            code_distance = self.jaccard(current_check_matrix, self.target_check_matrix) 
        elif self.distance_metric == 'hamming':
            code_distance = self.hamming(current_check_matrix, self.target_check_matrix)    

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
        '''
        Gaussian elimination to get canonical stabilizers
        
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
        new_state = jnp.matmul(state.tableau, self.actions[action]) % 2
        new_sign = self.update_signs(state.tableau, new_state, state.sign, action)

        new_propagated_error = self.update_propagated_errors(state.propagated_error, action, state.number_of_errors).to_numpy()

        new_number_of_errors = state.number_of_errors + 15

        current_flagged_errors = self.count_flagged_errors(new_state, new_propagated_error, new_number_of_errors)
        current_product_ancilla = self.check_ancilla_product_state(new_state, new_sign)
        current_distance = self.get_distance(new_state, new_sign)


        reward = self.weight_ancillas * (current_product_ancilla - state.previous_product_ancilla) +  \
                        self.weight_flag * (current_flagged_errors - state.previous_flagged_errors) +     \
                        self.weight_distance * (current_distance - state.previous_distance)
            
        state = EnvState(new_state.astype(jnp.uint8), new_sign.astype(jnp.uint8),  new_propagated_error.astype(jnp.uint8), current_flagged_errors, current_distance, current_product_ancilla, new_number_of_errors, state.time + 1)

        # Evaluate termination conditions
        done = self.is_terminal(state)


        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,            
            {"discount": self.discount(state, params)}
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
        tableaus = TableauSimulator(self.n_qubits_physical)

        ## Hardcode qubits into +
        for ii in self.plus_ancilla_position:
            tableaus.h(ii) 
 
        signs = tableaus.current_signs[0]
        tableaus = tableaus.current_tableau[0]

        propagated_errors = self.initial_propagated_errors.copy() 

        previous_distance = self.get_distance(tableaus, signs)
        previous_product_ancilla = self.check_ancilla_product_state(tableaus, signs)

        state = EnvState(
            tableau = tableaus,
            sign = signs,
            propagated_error = propagated_errors.to_numpy(),
            previous_flagged_errors=1.,
            previous_distance=previous_distance,
            previous_product_ancilla=previous_product_ancilla,
            number_of_errors = 0,
            time = 0
        )

        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """Applies observation function to state.
        Args:
            state: The state.
            params: The parameters.
        
        Return:
            Observations by appending the tableau and the sign
        """
        obs_tab, obs_sign = self.canonical_stabilizers(self.get_observation(state.tableau), state.sign[self.n_qubits_physical:] * 2)
        return jnp.append(obs_tab.flatten(), obs_sign // 2) 

    def is_terminal(self, state: EnvState, params=None) -> bool:
        """Check whether state is terminal.
        Args:
            state: The state.
            params: The parameters.
        
        Return:
            True if the distance is more than threshold or the time is more than max_steps.
        """
        # Check termination criteria
        check = (state.previous_flagged_errors + state.previous_distance + state.previous_product_ancilla) / 3.

        done_encoding = check > self.threshold
        
        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps
        
        done = jnp.logical_or(done_encoding, done_steps)
        return done

    def copy(self):
        """ Copy environment. """
        return FTLogicalStatePreparationEnv(
            self.target,
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
            self.group_ancillas,  
            self.cz_ancilla_only,
            self.plus_ancilla_position,
            self.distance_metric
            )

    @property
    def name(self) -> str:
        """Environment name."""
        return "FTLogicalStatePreparationEnv"

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