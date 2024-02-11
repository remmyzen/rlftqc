import jax.numpy as jnp

class CliffordGates():
    """ Clifford gates class that consists of Clifford gates to update tableau.
    Currently only support the following gates: H, S, X, SQRT_X, CX, and CZ.
    """
    
    def __init__(self, n):
        
        self.n = n # Number of qubits

    def h(self, i):
        """ Hadamard gate.
        Args:
            i (int): The qubit position.
        Returns:
            The operator to update tableau and the sign operator to update the sign
        """
        # Hadamard rule: X -> Z, Z -> X, Y -> -Y. For qubit i, this means that I swap columns i and i+n
        
        h_operator = jnp.eye(2*self.n, dtype=jnp.uint8)
        
        # Swap columns i and i+n
        temp = h_operator[:,i].copy()
        h_operator = h_operator.at[:,i].set(h_operator[:,i+self.n])
        h_operator = h_operator.at[:,i+self.n].set(temp)
        
        # Build a projector that selects columns i and i+n from the tableau
        projector = jnp.zeros((2*self.n, 2), dtype=jnp.uint8)
        projector = projector.at[i,0].set(1)
        projector = projector.at[i+self.n,1].set(1)
        # print(projector)
        
        # A multiplies column i by column i+n. A sign flip occurs when this product is 1
        # This means that there is a Y in that position
        A = jnp.array([[0,1],[0,0]])
        h_sign_operator = projector @ A @ projector.T
        
        # This will be multiplied as tableau @ h_sign_operator @ tableau.T
        # The diagonal elements of this operation are those products
        
        # Return the matrix representation
        # We return two copies of the sign operator for compatibility with 2-qubit gates
        return h_operator, jnp.array([h_sign_operator, h_sign_operator])


    def s(self,i):
        """ Phase gate.
        Args:
            i (int): The qubit position.
        Returns:
            The operator to update tableau and the sign operator to update the sign
        """
        # Phase gate rule: X -> Y, Z -> Z, Y -> -X.
        # For the operator, only need to make the Y generators at qubit i to Y.
        s_operator = jnp.eye(2*self.n, dtype=jnp.uint8)

        ## Make qubit i into Y 
        s_operator = s_operator.at[i, self.n + i ].set(1)

        # Build a projector that selects columns i and i+n from the tableau
        projector = jnp.zeros((2*self.n, 2), dtype=jnp.uint8)
        projector = projector.at[i,0].set(1)
        projector = projector.at[i+self.n,1].set(1)
        
        # A multiplies column i by column i+n. A sign flip occurs when this product is 1
        # This means that there is a Y in that position
        A = jnp.array([[0,1],[0,0]])
        s_sign_operator =  projector @ A @ projector.T
        
        # This will be multiplied as tableau @ s_sign_operator @ tableau.T
        # The diagonal elements of this operation are those products
        
        # Return the matrix representation
        # We return two copies of the sign operator for compatibility with 2-qubit gates
        return s_operator, jnp.array([s_sign_operator, s_sign_operator])

    def x(self,i):
        """ X gate.
        Args:
            i (int): The qubit position.
        Returns:
            The operator to update tableau and the sign operator to update the sign
        """
        # X gate rule: X -> X, Z -> -Z, Y -> -Y.
        # For the operator, only need to make the Z generators at qubit i to Y.
        x_operator = jnp.eye(2*self.n, dtype=jnp.uint8)

        # Build a projector that selects columns i and i+n from the tableau
        projector = jnp.zeros((2*self.n, 2), dtype=jnp.int8)
        
        projector = projector.at[i,0].set(1)
        projector = projector.at[i+self.n,0].set(1)
        projector = projector.at[i,1].set(0)
        projector = projector.at[i+self.n,1].set(1)

        # A multiplies column i by column i+n. A sign flip occurs when this product is 1
        # This means that there is a Y in that position
        A = jnp.array([[0,0],[0,1]])

        x_sign_operator =  projector @ A @ projector.T
    

        # This will be multiplied as tableau @ s_sign_operator @ tableau.T
        # The diagonal elements of this operation are those products
        
        # Return the matrix representation
        # We return two copies of the sign operator for compatibility with 2-qubit gates
        return x_operator, jnp.array([x_sign_operator, x_sign_operator])
        

    def sqrt_x(self,i):
        """ Square root of x gate.
        Args:
            i (int): The qubit position.
        Returns:
            The operator to update tableau and the sign operator to update the sign
        """
        # SQRT X gate rule: X -> X, Z -> -Y, Y -> Z .
        # For the operator, only need to make the Z generators at qubit i to Y.

        sqrt_x_operator = jnp.eye(2*self.n, dtype=jnp.uint8)

        ## Make qubit i into Y 
        sqrt_x_operator = sqrt_x_operator.at[i + self.n, i ].set(1)

        # Build a projector that selects columns i and i+n from the tableau
        projector = jnp.zeros((2*self.n, 2), dtype=jnp.uint8)
        projector = projector.at[i,0].set(1)
        # projector = projector.at[i+self.n,1].set(1)


        # projector = projector.at[i+self.n,0].set(1)
        projector = projector.at[i,1].set(1)
        projector = projector.at[i+self.n,1].set(1)

        # print(projector)
        # A multiplies column i by column i+n. A sign flip occurs when this product is 1
        # This means that there is a Y in that position
        A = jnp.array([[0,0],[1,1]])
        # A = jnp.array([[0,1],[0,0]])
        sqrt_x_sign_operator =  projector @ A @ projector.T
        
        # print(sqrt_x_sign_operator)

        # This will be multiplied as tableau @ s_sign_operator @ tableau.T
        # The diagonal elements of this operation are those products
        
        # Return the matrix representation
        # We return two copies of the sign operator for compatibility with 2-qubit gates
        return sqrt_x_operator, jnp.array([sqrt_x_sign_operator, sqrt_x_sign_operator])
        
        
    def cx(self, control, target):
        """ CNOT or CX gate.
        Args:
            control (int): The control qubit position.
            target (int): The target qubit position.
        Returns:
            The operator to update tableau and the sign operator to update the sign
        """
        # CX rule: X(c) -> X(c)X(t), Z(c) -> Z(c), X(t) -> X(t), Z(t) -> Z(c)Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]
        
        cx_operator = jnp.eye(2*self.n, dtype=jnp.uint8)
        
        # Transform X(c) -> X(c)X(t)
        cx_operator = cx_operator.at[control, target ].set(1)
        
        # Transform Z(t) -> Z(c)Z(t)
        cx_operator = cx_operator.at[target+self.n, control+self.n ].set(1)

        # Build two projectors: one selects columns control and control+n and the other one selects target and target+n from the tableau
        projector_control = jnp.zeros((2*self.n, 2), dtype=jnp.int8)
        projector_target = jnp.zeros((2*self.n, 2), dtype=jnp.int8)
    
        projector_control = projector_control.at[control,0].set(1)
        projector_control = projector_control.at[control+self.n,1].set(1)
        
        projector_target = projector_target.at[target,0].set(1)
        projector_target = projector_target.at[target+self.n,1].set(1)
        
        # A multiplies column i by column i+n. If 1, there is a Y in that position
        # The only flip is YY -> -XZ (but also XZ -> -YY)
        # Combining the control and the target sign operators will be done by the RL environment
        A = jnp.array([[0,1],[0,0]])
        control_sign_operator = projector_control @ A @ projector_control.T
        target_sign_operator = projector_target @ A @ projector_target.T

        # Return the matrix representation
        return cx_operator, jnp.array([control_sign_operator, target_sign_operator])
        
    
    def cz(self, control, target):
        """ CZ gate.
        Args:
            control (int): The control qubit position.
            target (int): The target qubit position.
        Returns:
            The operator to update tableau and the sign operator to update the sign
        """
        # CZ rule: X(c) -> X(c)Z(t), Z(c) -> Z(c), X(t) -> Z(c)X(t), Z(t) -> Z(t)
        # 1: column[t] -> column[c] + column[t]
        # 2: column[c+n] -> column[c+n] + column[t+n]
        
        cz_operator = jnp.eye(2*self.n, dtype=jnp.uint8)
        
        # Transform X(c) -> X(c)Z(t)
        cz_operator = cz_operator.at[control, target+self.n ].set(1)
        
        # Transform X(t) -> Z(c)X(t)
        cz_operator = cz_operator.at[target, control+self.n ].set(1)
        
        ## Build two projectors: one selects columns control and control+n and the other one selects target and target+n from the tableau
        projector_control = jnp.zeros((2*self.n, 2), dtype=jnp.int8)
        projector_target = jnp.zeros((2*self.n, 2), dtype=jnp.int8)
        
        projector_control = projector_control.at[control,0].set(1)
        projector_control = projector_control.at[control+self.n,1].set(1)
        
        projector_target = projector_target.at[target,0].set(1)
        projector_target = projector_target.at[target+self.n,1].set(1)
        
        # A multiplies column i by column i+n. If 1, there is a Y in that position
        # The only flip is XY -> -YX (but also YX -> -XY)
        # Combining the control and the target sign operators will be done by the RL environment
        A_control = jnp.array([[1,0],[1,0]]) ## Get X in the control
        A_target = jnp.array([[0,1],[0,0]]) ## Get Y in the traget
        control_sign_operator = projector_control @ A_control @ projector_control.T
        target_sign_operator = projector_target @ A_target @ projector_target.T

        # Return the matrix representation
        return cz_operator, jnp.array([control_sign_operator, target_sign_operator])