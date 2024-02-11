# rlftqc - Reinforcement Learning for Fault-Tolerant Quantum Circuit Discovery

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)

Code repository for quantum circuit discovery for fault-tolerant logical state preparation with reinforcement learning. 

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
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

## Usage


## License

MIT License

## Citation
```
@article{zen2024quantum,
  title={Quantum Circuit Discovery for Fault-Tolerant Logical State Preparation with Reinforcement Learning},
  author={},
  journal={arXiv:2402.xxxx},
  year={2024}
}
```


## References
[1] Chamberland, Christopher, and Michael E. Beverland. "Flag fault-tolerant error correction with arbitrary distance codes." Quantum 2 (2018): 53.

