from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
from flax import struct
import chex
from inspect import signature
from typing import Tuple, Optional

from rlftqc.simulators import PauliString, TableauSimulator, CliffordGates


@struct.dataclass
class EnvState:
    """ This class will contain the state of the environment.
    """
    tableau: jnp.array
    sign: jnp.array
    previous_distance: float
    time: int
    
@struct.dataclass
class EnvParams:
    """ This class will contain the default parameters of the environment.
    """
    n: int = 7
    k: int = 1
    max_steps_in_episode: int = 50

class LogicalStatePreparationEnv(environment.Environment):
    """Environment for the logical state preparation task.

    Args:
        target (list(str)): List of stabilizers of the target state as a string.
        gates (list(CliffordGates), optional): List of clifford gates to prepare the state. Default: H, S, CX. 
        graph (list(tuple), optional): Graph of the qubit connectivity. Default: all-to-all qubit connectivity.
        distance_metric (str, optional): Distance metric to use for the complementary distance reward.
            Currently only support 'hamming' or 'jaccard' (default).
        max_steps (int, optional): The number of maximum gates to be applied in the circuit. Default: 50
        threshold (float, optional): The complementary distance threshold to indicates success. Default: 0.99
        initialize_plus (list(int), optional): Initialize qubits given in the list as plus state instead of zero state.
            This is useful for large CSS codes or CZ is used in the gate set.
    """
    def __init__(self,
        target,       
        gates=None,   
        graph=None,        
        distance_metric = 'jaccard',
        max_steps = 50,
        threshold = 0.99,                 
        initialize_plus = []
        ):
        """Initialize a logical state preparation environment.
        """
        super().__init__()
        self.distance_metric = distance_metric
        self.max_steps = max_steps
        self.threshold = threshold

        ### Process target stabilizers
        self.target = target
        self.target_sign = []
        self.n_qubits_physical = len(self.target)
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

            if len(stab_wo_sign) != self.n_qubits_physical:
                raise ValueError("The length of stabilizer",stabs, "is not correct.")

            if not all(char in ['I', 'X', 'Y', 'Z'] for char in stab_wo_sign):
                raise ValueError("Stabilizer contains Pauli string other than I, X, Y, or Z.")

            target_wo_sign.append(stab_wo_sign)
        self.target_tableau = PauliString(target_wo_sign)

        ## Get canonical tableau
        self.target_check_matrix_unflatten = jnp.array(self.target_tableau.to_numpy())
        canon_target_check_matrix, canon_target_sign = self.canonical_stabilizers(self.target_check_matrix_unflatten, jnp.array(self.target_sign) * 2)
        self.canon_target_check_matrix = canon_target_check_matrix
        self.canon_target_sign = canon_target_sign
        self.target_check_matrix = jnp.append(canon_target_check_matrix.flatten(), canon_target_sign // 2)

        ## Initial Tableau
        self.initial_tableau = TableauSimulator(self.n_qubits_physical)
        self.initialize_plus = initialize_plus


        ## Process gates
        self.gates = gates
        if self.gates is None:
            clifford_gates = CliffordGates(self.n_qubits_physical)
            ## Initialize with the standard gate set
            self.gates = [clifford_gates.h, clifford_gates.s, clifford_gates.cx] 

        ### Generate graph
        self.graph = graph
        if self.graph is None:
            self.graph = []
            ## All-to-all qubit connectivity
            for ii in range(self.n_qubits_physical):
                for jj in range(ii+1, self.n_qubits_physical):
                    self.graph.append((ii,jj))
                    self.graph.append((jj,ii))


        ## Get observation shape and actions                                      
        self.obs_shape = self.get_observation(self.initial_tableau.current_tableau[0]).flatten().shape[0] + self.n_qubits_physical ## For the sign
        self.actions = self.action_matrix()

    def get_observation(self, tableau):
        """ Extract the check matrix for the observation of the RL agent.

        Args:
            tableau: check matrix of the tableau.

        Returns:
            Returns the stabilizer part of the tableau and ignore the destabilizers.
        """   
        check_mat = tableau[self.n_qubits_physical:].astype(jnp.uint8)
        return check_mat

    def action_matrix(self,
                      params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """ Generate the action matrix.

        Args:
            params (optional): Parameters of the environment

        Returns:
            Action matrix for each gate.
        """
        action_matrix = []
        self.action_string = []
        self.action_string_stim = []
        self.action_string_stim_circ = []
        self.sign_matrix = []
        self.sign_mask = []

        for gate in self.gates:
            ## One qubit gate
            if len(signature(gate).parameters) == 1:
                for n_qubit in range(self.n_qubits_physical):                    
                    action_matrix.append(gate(n_qubit)[0])
                    self.sign_matrix.append(gate(n_qubit)[1])                    
                    self.sign_mask.append(0)


                    self.action_string.append('%s-%d' % (gate.__name__, n_qubit))
                    self.action_string_stim.append('.%s(%d)' % (gate.__name__.lower(), n_qubit))
                    self.action_string_stim_circ.append('.append("%s", [%d])' % (gate.__name__.lower(), n_qubit))

            ## Two qubit gates
            elif len(signature(gate).parameters) == 2:
                for edge in self.graph:
                    action_matrix.append(gate(edge[0], edge[1])[0])     
                    self.sign_matrix.append(gate(edge[0], edge[1])[1])

                    self.sign_mask.append(1)
                    self.action_string.append('%s-%d-%d' % (gate.__name__, edge[0], edge[1]))
                    self.action_string_stim.append('.%s(%d, %d)' % (gate.__name__.lower(), edge[0], edge[1]))
                    self.action_string_stim_circ.append('.append("%s", [%d, %d])' % (gate.__name__.lower(), edge[0], edge[1]))
             
        self.sign_matrix = jnp.array(self.sign_matrix, dtype=jnp.uint8)
        self.sign_mask = jnp.array(self.sign_mask, dtype=jnp.uint8)

        return jnp.array(action_matrix, dtype=jnp.uint8)
  
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
        canon_check_matrix, canon_sign = self.canonical_stabilizers(self.get_observation(tableaus), signs[self.n_qubits_physical:] * 2)
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
    
    def swap_matrix(self, pos1, pos2):
        """ Create swap matrix that swap row of position 1 (pos1) and position 2 (pos2) for gaussian elimination.

        Args:
            pos1(int): Position 1
            pos2(int): Position 2

        Returns:
            A matrix that applied to the tableau swap row of pos1 and pos2.
        """

        swap = jnp.eye(self.n_qubits_physical)
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

    def canonical_stabilizers(self, check_matrix_inp, sign_inp):
        """ Get canonical stabilizers.

        Args:
            check_matrix_inp: the check matrix input.
            sign_inp: the sign input.
        
        Returns:
            canonical check matrix, canonical sign.
        """
        
        def cond(state):
            return state[0] < state[1].shape[0]

        def body(state):
            min_pivot, check_matrix, sign, ii, column_index = state
            
            pivot = min_pivot
            
            check_column = check_matrix[:, column_index[ii]].astype(jnp.uint8)
            mask = jnp.array(range(check_matrix.shape[0]))
            mask = jnp.where(mask >= pivot, 1, 0).astype(jnp.uint8)

            ## Get pivot value
            max_value = jnp.max(check_column * mask, initial=0)
            pivot = jnp.argmax(check_column * mask) 

            ## Create the matrix to eliminate ones
            elim = jnp.eye(num_qubits)
            elim = elim.at[:,pivot].set(check_column)
            condition = max_value

            check_matrix = (condition * ((elim @ check_matrix) % 2)) + ((1- condition) * check_matrix)

            ## Update sign
            check_column = check_column.at[pivot].set(0)
            repeat_pivot = jnp.repeat(pivot, num_qubits)
            sign = sign + (condition * check_column * ((sign[repeat_pivot] + self.ipow(check_matrix, check_matrix[repeat_pivot])) )) 
            sign = sign % 4


            ## Create swap matrix to swap columns
            condition = (min_pivot != pivot).astype(jnp.uint8) * max_value
            swap = self.swap_matrix(min_pivot, pivot)
            check_matrix = (condition * swap @ check_matrix) + (1-condition) * check_matrix
            sign = (condition * swap @ sign) + (1-condition) * sign


            min_pivot += 1 * max_value
            ii += 1
            return min_pivot, check_matrix.astype(jnp.uint8), sign.astype(jnp.uint8), ii, column_index

        num_qubits = check_matrix_inp.shape[0]
        column_index = jnp.zeros((2*num_qubits), dtype=jnp.uint8)
        column_index = column_index.at[0::2].set(jnp.array(range(num_qubits)))
        column_index = column_index.at[1::2].set(column_index[0::2] + num_qubits)
            
        init_val = (0, check_matrix_inp.astype(jnp.uint8), sign_inp.astype(jnp.uint8), 0, column_index)
        returned_state = jax.lax.while_loop(cond_fun = cond, body_fun=body, init_val= init_val)
        return returned_state[1], returned_state[2]
  
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
        ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
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
        new_sign = self.update_signs(state.tableau, new_state, state.sign, action)
        current_distance = self.get_distance(new_state, new_sign)

        ## Compute complementary distance reward
        reward = current_distance - state.previous_distance 

        state = EnvState( new_state, new_sign, current_distance, state.time + 1)

        # Evaluate termination conditions
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment.

        Args:
            key: Random key for Jax.
            params: Parameters.

        Returns: 
            observation, state.
        """
        tableaus = TableauSimulator(self.n_qubits_physical)

        for ii in self.initialize_plus: ## only data qubits
            tableaus.h(ii)   
                
        signs = tableaus.current_signs[0]
        tableaus = tableaus.current_tableau[0]

        state = EnvState(
            tableau = tableaus,
            sign = signs,            
            previous_distance=self.get_distance(tableaus, signs),
            time = 0
        )
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal.
        
        Args:
            state: The state.
            params: The parameters.
        
        Return:
            True if the distance is more than threshold or the time is more than max_steps.
        """
        # Check termination criteria
        done_encoding = state.previous_distance > self.threshold
        
        # Check number of steps in episode termination condition
        done_steps = state.time >= self.max_steps

        done = jnp.logical_or(done_encoding, done_steps)

        return done

    def get_obs(self, state: EnvState, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """Applies observation function to state.

        Args:
            state: The state.
            params: The parameters.
        
        Returns:
            Observations by appending the tableau and the sign
        """
        obs_tab, obs_sign = self.canonical_stabilizers(self.get_observation(state.tableau), state.sign[self.n_qubits_physical:] * 2)
        return jnp.append(obs_tab.flatten(), obs_sign // 2) 
  
    def __str__(self):
        '''Text representation.'''
        return 'LogicalStatePreparationEnv(target=%s, gates=%s, graph=%s)' % (str(self.target), str(self.gates), str(self.graph))

    def __repr__(self):
        '''Text representation.'''
        return 'LogicalStatePreparationEnv(target=%s, gates=%s, graph=%s)' % (str(self.target), str(self.gates), str(self.graph))

    def copy(self):
        """ Copy environment. """
        return LogicalStatePreparationEnv(self.target, self.gates, self.graph, self.distance_metric, self.max_steps, self.threshold, self.initialize_plus)

    @property
    def name(self) -> str:
        """Environment name."""
        return "LogicalStatePreparation-v0"

    @property
    def num_actions(self, params: Optional[EnvParams] = EnvParams) -> int:
        """Number of actions possible in environment."""
        return self.actions.shape[0]

    def action_space(
        self, params: Optional[EnvParams] = EnvParams
    ) -> spaces.Discrete:
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


