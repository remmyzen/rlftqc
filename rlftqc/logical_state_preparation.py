from rlftqc.envs.logical_state_preparation_env import LogicalStatePreparationEnv
import os 
import jax
from rlftqc.agents import make_train, ActorCritic
from rlftqc.utils import convert_stim_to_qiskit_circuit
import time 
import matplotlib.pyplot as plt
import json
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import stim

class LogicalStatePreparation:
    """ Class for training RL agent for logical state preparation task.
    """

    def __init__(self,
        target,
        gates=None,   
        graph=None,        
        distance_metric = 'jaccard',
        max_steps = 50,
        threshold = 0.99,                 
        initialize_plus = [],
        training_config = None,
        seed = 42):
        """
        Initialize a logical state preparation task.

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
            training_config (optional): Training configuration.
            seed (int, optional): Random seed (default: 42) 
        """

        ## Initialize the environment
        self.env = LogicalStatePreparationEnv(target, gates, graph, distance_metric, max_steps, threshold, initialize_plus)

        self.seed = seed

        ## Get the agent
        self.training_config = training_config
        if self.training_config is None:
            self.training_config = {
                "LR": 1e-3,
                "NUM_ENVS": 16,
                "NUM_STEPS": max_steps,
                "TOTAL_TIMESTEPS": 5e5,
                "UPDATE_EPOCHS": 4,
                "NUM_MINIBATCHES": 4,
                "GAMMA": 0.99,
                "GAE_LAMBDA": 0.95,
                "CLIP_EPS": 0.2,
                "ENT_COEF": 0.05,
                "VF_COEF": 0.5,
                "MAX_GRAD_NORM": 0.5,
                "ACTIVATION": "relu",
                "ANNEAL_LR": True,
                "NUM_AGENTS": 1,
            }


    def train(self):
        """ Training the agent.
        """
        #### Training
        rng = jax.random.PRNGKey(self.seed)
        rngs = jax.random.split(rng, self.training_config['NUM_AGENTS'])
        train_vjit = jax.jit(jax.vmap(make_train(self.training_config, self.env)))
        t0 = time.perf_counter()
        print("==== Training begin")
        self.outs = jax.block_until_ready(train_vjit(rngs))
        self.total_time = time.perf_counter() - t0
        print("==== Training finished, time elapsed: %.5f s" % (self.total_time))

    def run(self, results_folder_name=None):
        """ Run the trained agent.
        """        
        assert hasattr(self, 'outs'), "Call train() first to train the model."

            
        if results_folder_name is None:
            results_folder_name = 'results-lsp/%d/' % (self.env.n_qubits_physical)

        if not os.path.exists(results_folder_name):
            os.makedirs(results_folder_name)

        train_state, env_state, last_obs, rng = self.outs['runner_state']
        success_length = []
        success = 0

        for index in range(self.training_config['NUM_AGENTS']):
            ## Create a copy of parameter of the network manually
            params_new = {}
            params_new['params'] = {}
            for key in train_state.params['params'].keys():
                params_new['params'][key] = {} 
                for key2 in train_state.params['params'][key].keys():
                    params_new['params'][key][key2]= train_state.params['params'][key][key2][index]
            
            
            eval_env = self.env.copy()

            env_params = None
            network = ActorCritic(eval_env.action_space().n, activation=self.training_config["ACTIVATION"])
            rng = jax.random.PRNGKey(self.seed)
            reset_rng = jax.random.split(rng, self.training_config["NUM_ENVS"])
            obsv, env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, None)
            
            done = False
            actions = []
            length = 0
            while not np.any(done):   
                length += 1 
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(params_new, obsv)
                action = jnp.argmax(nn.softmax(pi.logits), 1)
                actions.append(int(action[0]))
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, self.training_config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(eval_env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
            
            if length < eval_env.max_steps:

                length += len(eval_env.initialize_plus)
                success += 1
                success_length.append(length)
                circ = stim.Circuit()

                for x_pos in eval_env.initialize_plus:
                    circ.append("H", [x_pos])

                for action in actions:
                    eval('circ%s' % eval_env.action_string_stim_circ[action])

                ## Save circuit
                circ.to_file('%s/circuit_%d.stim' % (results_folder_name, index))

                qc = convert_stim_to_qiskit_circuit(circ)
                qc.draw('mpl', filename='%s/circuit_%d.png' % (results_folder_name, index))

                ## Draw Circuit
                print("Circuit %d" % index)
                fig, ax = plt.subplots()
                qc.draw('mpl', ax=ax)
                plt.show()
                plt.close()
        
        if success == 0:
            print("==== No circuits found.")
            print("Tips: ")
            print("(1) Try to change the max_steps parameter to increase the number of possible gates in the circuit.")
            print("(2) Change the training configuration by increasing the TOTAL_TIMESTEPS to train longer. For example: training_config['TOTAL_TIMESTEPS'] = 1e6.")
            print("(3) Change the other training configurations.")
        else:
            print("==== Circuits saved in folder: ", results_folder_name)


    def log(self, results_folder_name=None):
        """ Log and visualize the training progress.
        Args:
            results_folder_name (str): Name of the results folder.
        """
        ""
        assert hasattr(self, 'outs'), "Call train() first to train the model."

                    
        if results_folder_name is None:
            results_folder_name = 'results-lsp/%d/' % (self.env.n_qubits_physical)

        if not os.path.exists(results_folder_name):
            os.makedirs(results_folder_name)

        print("==== Logging results in folder: ", results_folder_name)

        ## Save json configuration


        ## Log Time
        with open('%s/time.txt' % (results_folder_name), 'w') as f:
            f.write('%.5f\n' % (self.total_time))
        
        ## Training config
        with open('%s/training_config.json' % (results_folder_name), 'w') as f:
            json.dump(self.training_config, f)

        ## Environment
        with open('%s/env_config.txt' % (results_folder_name), 'w') as f:
            f.write(str(self.env) + '\n')

        ## Figures

        # Plot return
        fig = plt.figure()
        returns = []
        for i in range(self.training_config["NUM_AGENTS"]):
            plt.plot(self.outs["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1))
            returns.append(self.outs["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1))
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.show()

        pickle.dump(returns, open('%s/returns.p' % results_folder_name, 'wb'))
        fig.savefig('%s/return.png' % (results_folder_name), format='png', bbox_inches='tight')
        plt.close()

        # Plot lengths
        fig = plt.figure()
        lengths = []

        for i in range(self.training_config["NUM_AGENTS"]):
            plt.plot(self.outs["metrics"]["returned_episode_lengths"][i].mean(-1).reshape(-1))
            lengths.append(self.outs["metrics"]["returned_episode_lengths"][i].mean(-1).reshape(-1))
        plt.xlabel("Update Step")
        plt.ylabel("Circuit size")
        plt.show()

        pickle.dump(lengths, open('%s/lengths.p' % results_folder_name, 'wb'))

        fig.savefig('%s/length.png' % (results_folder_name), format='png', bbox_inches='tight')
        plt.close()        



