import jax.numpy as jnp

class TableauSimulator():
    """ A tableau simulator of batches Clifford circuits in parallel in Jax. 

    Args:
        n (int): Number of qubits.
        batch_size (int): Number of parallel circuits to simulate. Default: 1.
        initial_tableau (optional): Initial tableau in case one does not want to start from scratch. 
        initial_sign (optional): Initial sign of the tableau in case one does not want to start from scratch.
    """
    
    def __init__(self,
                 n,
                 batch_size = 1,
                 initial_tableau = None,
                 initial_sign = None):
        """ Initialized a tableau simulator.
        """
        
        self.n = n
        if initial_tableau is None:
            self.current_tableau = jnp.tile(jnp.eye(2*n), (batch_size, 1, 1)).astype(jnp.uint8) # Initialize the tableau with an empty circuit
        else:
            self.current_tableau = jnp.tile(initial_tableau, (batch_size, 1, 1)).astype(jnp.uint8) # Initialize the tableau from another tableau

        if initial_sign is None:
            self.current_signs = jnp.zeros((batch_size, 2*n)).astype(jnp.uint8) # Empty circuit with all |0> has + signs
        else:
            self.current_signs = jnp.tile(initial_sign, (batch_size, 1)).astype(jnp.uint8) # Initialize the sign from another tableau

        self.batch_size = batch_size
        
    def h(self, qubit):
        """To apply Hadamard to the tableau to initialize some qubits to plus state instead of zero.

        Args:
            qubit (int): Qubit position to apply Hadamard. 
        """
        # Hadamard rule: X -> Z, Z -> X. For qubit i, this means that I swap columns i and i+n
        
        h_operator = jnp.eye(2*self.n, dtype=jnp.uint8)
        
        # Swap columns i and i+n
        temp = h_operator[:,qubit].copy()
        h_operator = h_operator.at[:,qubit].set(h_operator[:,qubit+self.n])
        h_operator = h_operator.at[:,qubit+self.n].set(temp)
        
        # Update the current tableau (only right-multiplication!)
        self.current_tableau =  self.current_tableau @ h_operator
        
    def __iter__(self):
        '''Iterate through all tableau.
        '''
        return TableauSimulatorIterator(self)

    def __str__(self):
        '''Text representation.
        '''
        return str(self.current_tableau)
    
    def __repr__(self):
        '''Text representation.
        '''
        return str(self.current_tableau)


class TableauSimulatorIterator:
   ''' Iterator class '''
   def __init__(self, tableau_simulator):
       self._current_tableau = tableau_simulator.current_tableau
       self._batch_size = tableau_simulator.batch_size
       # member variable to keep track of current index
       self._index = 0

   def __next__(self):
       ''''Returns the next value from team object's lists '''
       if self._index < self._batch_size:
           result = self._current_tableau[self._index]
           self._index +=1
           return result
       # End of Iteration
       raise StopIteration    