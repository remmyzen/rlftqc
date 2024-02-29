.. rlftqc documentation master file, created by
   sphinx-quickstart on Sun Feb 11 16:19:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#################
Reinforcement Learning for Fault-Tolerant Quantum Circuit Discovery (rlftqc)
#################

Welcome to the documentation for rlftqc!

This is a code repository for quantum circuit discovery for fault-tolerant logical state preparation with reinforcement learning.

See the paper here: |arXiv|

Description
-----------

This library can be used to train an RL agent for three different tasks:

1. **Logical State Preparation**: Prepare a logical state from a given stabilizer QEC code. 

2. **Verification Circuit Synthesis**: Prepare a verification circuit from a given logical state preparation circuit based on flag-qubit protocols [1] to make the state preparation fault-tolerant. 

3. **Integrated Fault-Tolerant Logical State Preparation**: Integrates the above two tasks to prepare a logical state fault-tolerantly.

For all the tasks, the user can specify the Clifford gate set and qubit
connectivity.


Contents:
=========

.. toctree::
   :maxdepth: 2

   installation
   minimal_examples
   cite
   modules
* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`


References
================
[1] Chamberland, Christopher, and Michael E. Beverland. “Flag fault-tolerant error correction with arbitrary distance codes.” Quantum 2 (2018): 53.


.. |arXiv| image:: https://img.shields.io/badge/arXiv-2402.17761-b31b1b.svg
   :target: https://arxiv.org/abs/2402.17761
