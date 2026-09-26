#################
Minimal Examples
#################

Logical State Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~

For the logical state preparation task, you only need to specify the
target stabilizers of your logical state.

For example, the code below will train an RL agent to prepare the
:math:`|0\rangle_L` of the 7-qubit Steane code. It uses :math:`H`,
:math:`S`, and :math:`CNOT` gates and all-to-all qubit connectivity by
default.

.. code:: python

   from rlftqc.logical_state_preparation import LogicalStatePreparation

   target = ["+ZZZZZZZ", "+ZIZIZIZ", "+XIXIXIX", "+IZZIIZZ", "+IXXIIXX", "+IIIZZZZ", "+IIIXXXX"]

   lsp = LogicalStatePreparation(target)
   lsp.train()   ## Train the agent
   lsp.run()     ## Run the agent to get the circuit

Refer to the notebook ``notebooks/01 - Logical State Preparation.ipynb``
for more advanced examples (e.g. change the gate set and qubit
connectivity).

Verification Circuit Synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the verification circuit synthesis task, you only need to specify
the encoding circuit as a ``stim.Circuit`` `(see
reference) <https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit>`__
or ``qiskit.QuantumCircuit`` `(see
reference) <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit>`__
instance.

For example, the code below will train an RL agent to synthesize a
verification circuit to fault-tolerantly prepare :math:`|0\rangle_L` of
the 7-qubit Steane code.

.. code:: python

   import stim
   from rlftqc.verification_circuit_synthesis import VerificationCircuitSynthesis

   ## Encoding circuit for the $|0\rangle_L$ of the 7-qubit Steane code.
   circ = stim.Circuit(""" H 0 1 3
   CX 0 6 1 5 0 4 3 4 3 5 5 6 0 2 1 2 """)

   ## We can ignore Z error since we are preparing zero-logical of Steane code
   vcs = VerificationCircuitSynthesis(circ, ignore_z_errors = True)  
   vcs.train()   ## Train the agent
   vcs.run()     ## Run the agent to get the circuit

Refer to the notebook
``notebooks/02 - Verification Circuit Synthesis.ipynb`` for more
advanced examples.

Integrated Fault-Tolerant Logical State Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the integrated logical state preparation task, you only need to
specify the target stabilizers of your logical state.

For example, the code below will train an RL agent to fault-tolerantly
prepare the :math:`|0\rangle_L` of the 7-qubit Steane code.

.. code:: python

   from rlftqc.ft_logical_state_preparation import FTLogicalStatePreparation

   target = ["+ZZZZZZZ", "+ZIZIZIZ", "+XIXIXIX", "+IZZIIZZ", "+IXXIIXX", "+IIIZZZZ", "+IIIXXXX"]

   ## We can ignore Z error since we are preparing zero-logical of Steane code
   ftlsp = FTLogicalStatePreparation(target, ignore_z_errors=True)
   ftlsp.train()   ## Train the agent
   ftlsp.run()     ## Run the agent to get the circuit

Refer to the notebook
``notebooks/03 - Integrated Fault-Tolerant Logical State Preparation.ipynb``
for more advanced examples. ## Circuit Examples

Go to this link to see the circuit examples that the RL agent has
synthesized for various tasks in PNG, stim, and Latex formats.