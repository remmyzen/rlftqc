# rlftqc - Reinforcement Learning for Fault-Tolerant Quantum Circuit Discovery

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)

Code repository for quantum circuit discovery for fault-tolerant logical state preparation with reinforcement learning. 

- [Description](#description)
- [Installation](#installation)
- [Minimal Examples](#minimal-examples)
- [License](#license)
- [Citation](#citation)
  
## Description

This library can be used to train an RL agent for three different tasks:
1. **Logical State Preparation**: Prepare a logical state from a given stabilizer QEC code.
2. **Verification Circuit Synthesis**: Prepare a verification circuit from a given logical state preparation circuit based on flag-qubit protocols [1] to make the state preparation fault-tolerant.
3. **Integrated Fault-Tolerant Logical State Preparation**: Integrates the two tasks above to prepare a logical state fault-tolerantly.

For all the tasks, the user can specify the Clifford gate set and qubit connectivity.

<img src="images/overview.png" alt="overview" width="800"/>


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
For logical state preparation, you only need to specify the target stabilizers of your logical state. 

For example, the code below will train an RL agent to prepare $|0\rangle_L$ of the 7-qubit Steane code. It uses $H$, $S$, and $CNOT$ gates and all-to-all qubit connectivity by default.

``` python
from rlftqc.logical_state_preparation import LogicalStatePreparation

target = ["+ZZZZZZZ", "+ZIZIZIZ", "+XIXIXIX", "+IZZIIZZ", "+IXXIIXX", "+IIIZZZZ", "+IIIXXXX"]

lsp = LogicalStatePreparation(target)
lsp.train()   ## Train the agent
lsp.run()     ## Run the agent to get the circuit
```

Refer to the notebook `notebooks/01 - Logical State Preparation.ipynb` for more advanced examples (e.g. change the gate set and qubit connectivity).

## License

The code in this repository is released under the MIT License.

## Citation
``` bib
@article{zen2024quantum,
  title={Quantum Circuit Discovery for Fault-Tolerant Logical State Preparation with Reinforcement Learning},
  author={},
  journal={arXiv:2402.xxxx},
  year={2024}
}
```

## References
[1] Chamberland, Christopher, and Michael E. Beverland. "Flag fault-tolerant error correction with arbitrary distance codes." Quantum 2 (2018): 53.

