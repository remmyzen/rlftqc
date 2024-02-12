import jax.numpy as jnp
## Jax does not support array of strings, use numpy to help process
import numpy as np
import time 
import more_itertools, functools
import jax.lax

class PauliString:
    '''Represents a batch of collection of Pauli operations (I, X, Y, Z) applied pairwise to a
    collection of qubits.

    No sign or phase for now.
    '''

    def __init__(self, *args):
        '''
            1. __init__(self, num_qubits: int, batch_size: int = 1) 
                Initialize IIII...I paulis with the size of batch_size.

            2. __init__(self, pauli_string: arr[str]) 
                Pauli String from arr of string.

            3. __init__(self) 
                Empty pauli string
        
        '''

        if len(args) == 0:
            self.index_matrix = None
            self.num_qubits = 0
            self.batch_size = 0
            self.check_matrix = None

        elif isinstance(args[0],int):
            self.num_qubits = args[0]
            if len(args) == 1:
                self.batch_size = 1
            else: 
                self.batch_size = int(args[1])
             
            self.index_matrix = jnp.zeros((self.batch_size, self.num_qubits), dtype=jnp.uint8)
            self.check_matrix = jnp.zeros((self.batch_size, self.num_qubits * 2), dtype=jnp.uint8)

        ## List of pauli strings
        elif isinstance(args[0],list):
            ## Jax does not support array of strings
            strings = np.array(args[0])
            self.index_matrix = self._string_to_index_matrix(strings)
            self.num_qubits = self.index_matrix.shape[1]
            self.batch_size = self.index_matrix.shape[0]
            self.check_matrix = self._index_to_check_matrix(self.index_matrix)

          
    def _string_to_index_matrix(self, strings):  
        '''
        Convert strings of pauli to index matrix where I = 0, X = 2, Y = 3, Z = 1
        Examples:
            ['XYZI', 'IIYX'] -> [[1,2,3,0], [0,0,2,1]]
        '''
        ## convert to array of characters  
        chars = strings.astype(bytes).view('S1').reshape((strings.size, -1))

        ## define mapping I = 0, X = 2, Y = 3, Z = 1
        chars = np.char.replace(chars, b'I', '0')
        chars = np.char.replace(chars, b'_', '0')
        chars = np.char.replace(chars, b'X', '2')
        chars = np.char.replace(chars, b'Y', '3')
        chars = np.char.replace(chars, b'Z', '1')
        return chars.astype(int)

    def _index_matrix_to_strings(self, index_matrix):
        '''
        Convert index matrix to strings of pauli where I = 0, X = 2, Y = 3, Z = 1
        Examples:
             [[1,2,3,0], [0,0,2,1]] -> ['XYZI', 'IIYX']
        '''
        ## convert to strings
        chars = index_matrix.astype(str)
        chars = np.char.replace(chars, '0', 'I')
        chars = np.char.replace(chars, '2', 'X')
        chars = np.char.replace(chars, '3', 'Y')
        chars = np.char.replace(chars, '1', 'Z')
        return np.array([''.join(row) for row in chars.tolist()])


    def _check_to_index_matrix(self, check_matrix):
        '''
        Convert check matrix to index matrix
        '''
        ##NOTE: Still using for loop

        index_matrix = jnp.zeros((check_matrix.shape[0], self.num_qubits), dtype=jnp.uint8)

        for ii in range(self.num_qubits):
            # index_matrix[:, ii] = 2 * check_matrix[:,ii] + check_matrix[:,ii+self.num_qubits]
            temp = 2 * check_matrix[:,ii] + check_matrix[:,ii+self.num_qubits]
            index_matrix = index_matrix.at[:, ii].set(temp)

        return index_matrix

    def _check_to_index_matrix_3d(self, check_matrix):
        '''
        Convert check matrix to index matrix
        '''
        ##NOTE: Still using for loop

        index_matrix = jnp.zeros((check_matrix.shape[0],check_matrix.shape[1], self.num_qubits), dtype=jnp.uint8)

        for ii in range(self.num_qubits):
            # temp =  2 * check_matrix[:,:,ii] + check_matrix[:,:,ii+self.num_qubits]
            # index_matrix = index_matrix.at[:,:, ii].set(temp)
            index_matrix = index_matrix.at[:,:, ii].set(2 * check_matrix[:,:,ii] + check_matrix[:,:,ii+self.num_qubits])
        return index_matrix


    def _index_to_check_matrix(self, index_matrix):
        '''
        Convert index matrix to check matrix.
        Examples:
            XYIZ  which is [2, 3, 0, 1] is [1,1,0,0,  0,1,0,1]
        '''
        ### No loop version, quite scalable and a bit faster than the old one
        ## Convert to binaries
        bits = jnp.unpackbits(jnp.array(index_matrix, dtype=jnp.uint8), axis=1)

        ## Reorder to make check matrix
        ## 8 is because of 8 bits, 6 and 7 is since we only care about the last two bits
        check_matrix = bits[:, list(range(6,8*self.num_qubits, 8)) + list(range(7,8*self.num_qubits, 8))]

        ## old version need to change mapping X = 0 and so on
        # Create zero check matrix
        # check_matrix = jnp.zeros((index_matrix.shape[0], self.num_qubits * 2), dtype=int)
        # for ii in range(index_matrix.shape[1]):
        #     ## set position of X pauli at ii to 1
        #     check_matrix[index_matrix[:,ii] == 1,ii] = 1
        #     ## set position of Y pauli at ii and ii + num_qubits to 1
        #     check_matrix[index_matrix[:,ii] == 2,ii] = 1
        #     check_matrix[index_matrix[:,ii] == 2,ii+self.num_qubits] = 1
        #     ## set position of Z pauli at ii + num_qubits to 1
        #     check_matrix[index_matrix[:,ii] == 3,ii+self.num_qubits] = 1

        return check_matrix
    
    def copy(self):
        '''
        Copy PauliString
        '''
        new_pauli = PauliString()
        new_pauli.index_matrix = jnp.copy(self.index_matrix)
        new_pauli.num_qubits = self.num_qubits
        new_pauli.batch_size = self.batch_size 
        new_pauli.check_matrix = jnp.copy(self.check_matrix)
        return new_pauli


    def update_pauli(self, another_pauli_string, position):
        '''
        Update pauli at the current position with another pauli string since Jax cannot handle dynamic array.
        '''
        self.check_matrix = jax.lax.dynamic_update_slice(self.check_matrix, another_pauli_string.check_matrix, (position,0))
        self.index_matrix = jax.lax.dynamic_update_slice(self.index_matrix, another_pauli_string.index_matrix, (position,0))
        # self.check_matrix = self.check_matrix.at[position:position + another_pauli_string.batch_size].set(another_pauli_string.check_matrix)
        # self.index_matrix = self.index_matrix.at[position:position + another_pauli_string.batch_size].set(another_pauli_string.index_matrix)

        return self

    def append(self, another_pauli_string):
        '''
        Append another pauli string
        '''
        # stack the matrix
        self.index_matrix = jnp.vstack((self.index_matrix, another_pauli_string.index_matrix))
        self.check_matrix = jnp.vstack((self.check_matrix, another_pauli_string.check_matrix))

        # remove duplicates
        # self.index_matrix = jnp.unique(self.index_matrix, axis=0)
        # self.check_matrix = jnp.unique(self.check_matrix, axis=0)


        # update batch size
        self.batch_size = self.index_matrix.shape[0]

        return self

    
    def insert_ancilla(self, num_ancillas):
        self.check_matrix = jnp.c_[self.check_matrix[:,:self.num_qubits], jnp.zeros((self.batch_size, num_ancillas)), self.check_matrix[:,self.num_qubits:], jnp.zeros((self.batch_size, num_ancillas))]
        self.index_matrix = jnp.c_[self.index_matrix, jnp.zeros((self.batch_size, num_ancillas))]
        self.num_qubits += self.num_ancillas
        return self

    def append_check_matrix(self, another_check_matrix):
        '''
        Append another pauli string
        '''
        # stack the matrix
        self.check_matrix = jnp.vstack((self.check_matrix, another_check_matrix))
        self.index_matrix = self._check_to_index_matrix(self.check_matrix)

        # remove duplicates
        # self.index_matrix = jnp.unique(self.index_matrix, axis=0)
        # self.check_matrix = jnp.unique(self.check_matrix, axis=0)


        # update batch size
        self.batch_size = self.index_matrix.shape[0]

        return self
    
    def after(self, gate):
        '''
        Update pauli after applying gate
        '''
        # multiply with gate from the right
        self.check_matrix = jnp.matmul(self.check_matrix, gate) % 2
        # update the index matrix
        self.index_matrix = self._check_to_index_matrix(self.check_matrix)

        # remove duplicates
        # self.index_matrix = jnp.unique(self.index_matrix, axis=0)
        # self.check_matrix = jnp.unique(self.check_matrix, axis=0)

        # update batch size
        self.batch_size = self.index_matrix.shape[0]

        return self
    
    def weight(self, index_matrix, index=None, ignore_x=False, ignore_y=False, ignore_z=False):
        if index is None:
            if ignore_x:
                return jnp.sum(index_matrix == 1, axis = 1) + jnp.sum(index_matrix == 3, axis = 1)
            elif ignore_y:
                return jnp.sum(index_matrix == 1, axis = 1) + jnp.sum(index_matrix == 2, axis = 1)
            elif ignore_z:
                return jnp.sum(index_matrix == 2, axis = 1) + jnp.sum(index_matrix == 3, axis = 1)
            else:
                return jnp.count_nonzero(index_matrix, axis=1)
        else:
            if ignore_x:
                return jnp.sum(index_matrix[:,index] == 1, axis = 1) + jnp.sum(index_matrix[:,index] == 3, axis = 1)
            elif ignore_y:
                return jnp.sum(index_matrix[:,index] == 1, axis = 1) + jnp.sum(index_matrix[:,index] == 2, axis = 1)
            elif ignore_z:
                return jnp.sum(index_matrix[:,index] == 2, axis = 1) + jnp.sum(index_matrix[:,index] == 3, axis = 1)
            else:
                return jnp.count_nonzero(index_matrix[:,index], axis=1)
    
    def weight_3d(self, index_matrix, index=None, ignore_x=False, ignore_y=False, ignore_z=False):
        if index is None:
            if ignore_x:
                return jnp.sum(index_matrix == 1, axis = 2) + jnp.sum(index_matrix == 3, axis = 2)
            elif ignore_y:
                return jnp.sum(index_matrix == 1, axis = 2) + jnp.sum(index_matrix == 2, axis = 2)
            elif ignore_z:
                return jnp.sum(index_matrix == 2, axis = 2) + jnp.sum(index_matrix == 3, axis = 2)
            else:
                return jnp.count_nonzero(index_matrix, axis=2)
        else:
            if ignore_x:
                return jnp.sum(index_matrix[:,index] == 1, axis = 2) + jnp.sum(index_matrix[:,index] == 3, axis = 2)
            elif ignore_y:
                return jnp.sum(index_matrix[:,index] == 1, axis = 2) + jnp.sum(index_matrix[:,index] == 2, axis = 2)
            elif ignore_z:
                return jnp.sum(index_matrix[:,index] == 2, axis = 2) + jnp.sum(index_matrix[:,index] == 3, axis = 2)
            else:
                return jnp.count_nonzero(index_matrix[:,index], axis=2)


    def get_weight(self, index=None, ignore_x=False, ignore_y=False, ignore_z=False):
        ''' 
        count the weight of the pauli
        '''
        return self.weight(self.index_matrix, index, ignore_x, ignore_y, ignore_z)

    def get_inverse_weight(self, index=None, include_x = False, include_y = False, include_z=False):
        ''' 
        count the inverse weight of the pauli (count only I or Z if include_z is True)
        '''
        if index is None:
            if include_z:
                return jnp.sum(self.index_matrix == 0, axis = 1) + jnp.sum(self.index_matrix == 1, axis = 1)
            elif include_x:
                return jnp.sum(self.index_matrix == 0, axis = 1) + jnp.sum(self.index_matrix == 2, axis = 1)
            elif include_y:
                return jnp.sum(self.index_matrix == 0, axis = 1) + jnp.sum(self.index_matrix == 3, axis = 1)
            else:
                return jnp.sum(self.index_matrix == 0, axis = 1)
        else:
            if include_z:
                return jnp.sum(self.index_matrix[:,index] == 0, axis = 1) + jnp.sum(self.index_matrix[:,index] == 1, axis = 1)
            elif include_x:
                return jnp.sum(self.index_matrix[:,index] == 0, axis = 1) + jnp.sum(self.index_matrix[:,index] == 2, axis = 1)
            elif include_y:
                return jnp.sum(self.index_matrix[:,index] == 0, axis = 1) + jnp.sum(self.index_matrix[:,index] == 3, axis = 1)
            else:
                return jnp.sum(self.index_matrix[:,index] == 0, axis = 1)

    def generate_stabilizer_group(self):
        '''
        Generate stabilizer group of the Pauli Strings.
        '''
        ## Generate powerset for all rows of the check matrix
        powerset = list(more_itertools.powerset(self.check_matrix))[1:]

        ## Multiply (sum modulo 2) all of the powerset element + IIIII...I
        new_check_matrix = jnp.array([jnp.zeros(self.num_qubits * 2)] + [functools.reduce(lambda x,y: (x+y) % 2, mat) for mat in powerset]).astype(jnp.uint8)

        return PauliString().from_numpy(new_check_matrix)
    
    def multiply_each(self):
        '''
        multiply with itself (use to generate combination of two qubit errors)
        TODO: still very inefficient
        '''
        import itertools
        check_matrix_new = []
        combinations = list(itertools.combinations(range(self.batch_size), 2))
        len_comb = len(combinations)
        print(len_comb)
        for ii, combs in enumerate(combinations):
            check_matrix_new.append((self.check_matrix[combs[0]] + self.check_matrix[combs[1]]) % 2)
        

        check_matrix_new = jnp.array(check_matrix_new).astype(jnp.uint8)
        new_paulistring = PauliString().from_numpy(check_matrix_new)
        return PauliString().from_numpy(check_matrix_new)


    def multiply(self, pauli):
        '''
        multiply with other paulis(use to generate combination of two qubit errors)
        '''
        ## Expand dimension to enable summation
        current_check_matrix = jnp.expand_dims(self.check_matrix, 1)
        pauli_check_matrix = jnp.expand_dims(pauli.check_matrix, 0)

        ## multiply is just summation modulo 2
        multiply = (current_check_matrix + pauli_check_matrix) % 2

        self.check_matrix = multiply.astype(jnp.uint8)
        self.check_matrix = self.check_matrix.reshape(self.check_matrix.shape[0] * self.check_matrix.shape[1], self.check_matrix.shape[2])
        # update the index matrix
        self.index_matrix = self._check_to_index_matrix(self.check_matrix)

        # remove duplicates
        # self.index_matrix = jnp.unique(self.index_matrix, axis=0)
        # self.check_matrix = jnp.unique(self.check_matrix, axis=0)

        # update batch size
        self.batch_size = self.index_matrix.shape[0]

        return self
            
    def multiply_and_update(self, pauli, num_ancillas, ignore_z, ignore_x=False, ignore_y = False):
        '''
        multiply with other paulis and update to smallest weight (use to multiply by generators)
        '''
        ## Expand dimension to enable summation
        current_check_matrix = jnp.expand_dims(self.check_matrix, 1)
        pauli_check_matrix = jnp.expand_dims(pauli.check_matrix, 0)

        ## multiply is just summation modulo 2
        multiply = (current_check_matrix + pauli_check_matrix) % 2
        
        ## FOR DEBUG
        # sum_index = jnp.array([self._check_to_index_matrix(a) for a in multiply])
        # sum_str = jnp.array([self._index_matrix_to_strings(a) for a in sum_index])
        # for a,b in zip(self._index_matrix_to_strings( self.index_matrix), sum_str):
        #     print('>>>', a, b)

        # print('>>>',sum_str)
        ######


        # get index matrix to count weight
        ##below is slower non-optimized version
        # sum_weight = jnp.array([self.weight(self._check_to_index_matrix(mult)[:,:-num_ancillas], ignore_x=ignore_x, ignore_y=ignore_y, ignore_z=ignore_z) for mult in multiply])
        sum_weight = self.weight_3d(self._check_to_index_matrix_3d(multiply)[:,:,:-num_ancillas], ignore_x=ignore_x, ignore_y=ignore_y, ignore_z=ignore_z)


        # put the check matrix minimal weight to the front and select them
        # NOTE: might be a better way
        argmin = jnp.argsort(sum_weight, axis = 1)
        argmin = jnp.repeat(argmin[:, :, jnp.newaxis], self.num_qubits * 2, axis=2)
        minimals = jnp.take_along_axis(multiply, argmin, axis=1)[:,0,:]

        self.check_matrix = minimals.astype(jnp.uint8)
        # update the index matrix
        self.index_matrix = self._check_to_index_matrix(self.check_matrix)

        # remove duplicates
        # self.index_matrix = jnp.unique(self.index_matrix, axis=0)
        # self.check_matrix = jnp.unique(self.check_matrix, axis=0)

        # update batch size
        self.batch_size = self.index_matrix.shape[0]

        return self
    
    def get_unique(self, size):
        new_pauli = PauliString()
        new_pauli.index_matrix = jnp.unique(self.index_matrix, axis = 0, size=size)
        new_pauli.num_qubits = self.num_qubits
        new_pauli.batch_size = size 
        new_pauli.check_matrix = jnp.unique(self.check_matrix, axis = 0, size=size)
        return new_pauli
        

    def remove_duplicate(self):
        '''
        Remove duplicated error, can only be done in non-jitted way.
        '''
        self.index_matrix = jnp.unique(self.index_matrix, axis=0)
        self.check_matrix = jnp.unique(self.check_matrix, axis=0)

        # update batch size
        self.batch_size = self.index_matrix.shape[0]

        return self
        
    def to_numpy(self):
        '''
        Return the check matrix
        '''
        return self.check_matrix
    
    def from_numpy(self, check_matrix):
        '''
        Construct tableau from a given check matrix
        '''
        self.check_matrix = check_matrix

        self.batch_size = self.check_matrix.shape[0]
        self.num_qubits = self.check_matrix.shape[1] // 2

        self.index_matrix = self._check_to_index_matrix(self.check_matrix)

        return self
    
    def get_pauli_strings(self):
        '''
        Get list of pauli strings
        '''
        return self._index_matrix_to_strings(self.index_matrix)

    def __getitem__(self, index_or_slice: int): 
        '''
        Access pauli according to index or slice
        '''
        return self.index_matrix[index_or_slice]

    def __str__(self):
        '''
        Text representation
        '''
        string = self._index_matrix_to_strings(self.index_matrix)
        return ','.join(["'%s'" % s for s in string])

    def __iter__(self):
        '''
        Iterate through all Paulis
        '''
        return PauliStringIterator(self)
    
    def __repr__(self):
        string = self._index_matrix_to_strings(self.index_matrix)
        return 'PauliString([%s])' % (','.join(["'%s'" % s for s in string]))





class PauliStringIterator:
   ''' Iterator class '''
   def __init__(self, paulis):
       self._strings = paulis._index_matrix_to_strings(paulis.index_matrix)
       # member variable to keep track of current index
       self._index = 0

   def __next__(self):
       ''''Returns the next value from team object's lists '''
       if self._index < self._strings.shape[0]:
           result = self._strings[self._index]
           self._index +=1
           return result
       # End of Iteration
       raise StopIteration    