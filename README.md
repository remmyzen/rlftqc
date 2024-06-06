# rlftqc - Reinforcement Learning for Fault-Tolerant Quantum Circuit Discovery

[![arXiv](https://img.shields.io/badge/arXiv-2402.17761-b31b1b.svg)](https://arxiv.org/abs/2402.17761)

Code repository for quantum circuit discovery for fault-tolerant logical state preparation with reinforcement learning. 

- [Description](#description)
- [Installation](#installation)
- [Minimal Examples](#minimal-examples)
    1. [Logical State Preparation](#logical-state-preparation)   <a href="https://colab.research.google.com/drive/1u2iokg1ZBF6YeB6-UuzmbFqAo_3KlCu-" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    2. [Verification Circuit Synthesis](#verification-circuit-synthesis)  <a href="https://colab.research.google.com/drive/1OJJ_DSpO7zUeoBZruXMIpntbWjXylVPf" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    3. [Integrated Fault-Tolerant Logical State Preparation](#integrated-fault-tolerant-logical-state-preparation)  <a href="https://colab.research.google.com/drive/1kcq8q0C1jE8J5xSVy19fpsnr0KdPTQwe" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [Documentations](https://remmyzen.github.io/rlftqc/)
- [Circuit Examples](#circuit-examples)
- [License](#license)
- [Citation](#citation)
  
## Description

This library can be used to train an RL agent for three different tasks:
1. **Logical State Preparation**: Prepare a logical state from a given stabilizer QEC code.
2. **Verification Circuit Synthesis**: Prepare a verification circuit from a given logical state preparation circuit based on flag-qubit protocols [1] to make the state preparation fault-tolerant.
3. **Integrated Fault-Tolerant Logical State Preparation**: Integrates the above two tasks to prepare a logical state fault-tolerantly.

For all the tasks, the user can specify the Clifford gate set and qubit connectivity. 

<img src="images/overview.png" alt="overview" width="800"/>

The implementation of reinforcement learning with a non-cumulative reward based on [2] is also possible by setting `use_max_reward = True` in the environments.

## Installation

1. Clone the repository

``` bash
git clone https://github.com/remmyzen/rlftqc.git
cd rlftqc
```

2. Install requirements
``` bash
pip install -r requirements.txt
```
## Minimal Examples

### Logical State Preparation  

For the logical state preparation task, you only need to specify the target stabilizers of your logical state. 

For example, the code below will train an RL agent to prepare the $|0\rangle_L$ of the 7-qubit Steane code. It uses $H$, $S$, and $CNOT$ gates and all-to-all qubit connectivity by default.

<a href="https://colab.research.google.com/drive/1u2iokg1ZBF6YeB6-UuzmbFqAo_3KlCu-" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

``` python
from rlftqc.logical_state_preparation import LogicalStatePreparation

target = ["+ZZZZZZZ", "+ZIZIZIZ", "+XIXIXIX", "+IZZIIZZ", "+IXXIIXX", "+IIIZZZZ", "+IIIXXXX"]

lsp = LogicalStatePreparation(target)
lsp.train()   ## Train the agent
lsp.run()     ## Run the agent to get the circuit
```

Refer to the notebook `notebooks/01 - Logical State Preparation.ipynb` <a href="https://drive.google.com/file/d/1EBmGK5bSTiSBJdnbAYyRLWfqJlXwm6NK/view?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for more advanced examples (e.g. change the gate set and qubit connectivity). 


### Verification Circuit Synthesis  


For the verification circuit synthesis task, you only need to specify the encoding circuit as a `stim.Circuit` [(see reference)](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit) or `qiskit.QuantumCircuit` [(see reference)](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit) instance. 

For example, the code below will train an RL agent to synthesize a verification circuit to fault-tolerantly prepare $|0\rangle_L$ of the 7-qubit Steane code.

<a href="https://colab.research.google.com/drive/1OJJ_DSpO7zUeoBZruXMIpntbWjXylVPf" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

``` python
import stim
from rlftqc.verification_circuit_synthesis import VerificationCircuitSynthesis

## Encoding circuit for the $|0\rangle_L$ of the 7-qubit Steane code.
circ = stim.Circuit(""" H 0 1 3
CX 0 6 1 5 0 4 3 4 3 5 5 6 0 2 1 2 """)

## We can ignore Z error since we are preparing zero-logical of Steane code
vcs = VerificationCircuitSynthesis(circ, ignore_z_errors = True)  
vcs.train()   ## Train the agent
vcs.run()     ## Run the agent to get the circuit
```

Refer to the notebook `notebooks/02 - Verification Circuit Synthesis.ipynb` <a href="https://drive.google.com/file/d/1gtI2cxYOsspWSHffVSsXry_0teQ_4d9b/view?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for more advanced examples. 


### Integrated Fault-Tolerant Logical State Preparation
  
For the integrated logical state preparation task, you only need to specify the target stabilizers of your logical state. 

For example, the code below will train an RL agent to  fault-tolerantly prepare the $|0\rangle_L$ of the 7-qubit Steane code. 

<a href="https://colab.research.google.com/drive/1kcq8q0C1jE8J5xSVy19fpsnr0KdPTQwe?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

``` python
from rlftqc.ft_logical_state_preparation import FTLogicalStatePreparation

target = ["+ZZZZZZZ", "+ZIZIZIZ", "+XIXIXIX", "+IZZIIZZ", "+IXXIIXX", "+IIIZZZZ", "+IIIXXXX"]

## We can ignore Z error since we are preparing zero-logical of Steane code
ftlsp = FTLogicalStatePreparation(target, ignore_z_errors=True)
ftlsp.train()   ## Train the agent
ftlsp.run()     ## Run the agent to get the circuit

```
Refer to the notebook `notebooks/03 - Integrated Fault-Tolerant Logical State Preparation.ipynb` <a href="https://drive.google.com/file/d/12zTTrUSPTK0dRym5XTizm2ugcZsoeki8/view?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
 for more advanced examples. 
## Circuit Examples

Go to this <a href="https://owncloud.gwdg.de/index.php/s/OsfE9WuvTitJuZv" target="_blank">link</a> to see the circuit examples that the RL agent has synthesized for various tasks in PNG, stim, and Latex formats.

## License

The code in this repository is released under the MIT License.

## Citation
``` bib
@article{zen_quantum_2024,
  title={Quantum Circuit Discovery for Fault-Tolerant Logical State Preparation with Reinforcement Learning},
  author={Zen, Remmy and Olle, Jan and Colmenarez, Luis and Puviani, Matteo and M{\"u}ller, Markus and Marquardt, Florian},
  url = {http://arxiv.org/abs/2402.17761},
  journal={arXiv preprint arXiv:2402.17761},
  urldate = {2024-02-27},
  publisher = {arXiv},
  month = feb,
  year = {2024},
  note = {arXiv:2402.17761 [quant-ph]},
}
```

## References
[1] Chamberland, Christopher, and Michael E. Beverland. "Flag fault-tolerant error correction with arbitrary distance codes." Quantum 2 (2018): 53.
[2] Nägele, M., Olle, J., Fösel, T., Zen, R., & Marquardt, F. (2024). Tackling Decision Processes with Non-Cumulative Objectives using Reinforcement Learning. arXiv:2405.13609.

