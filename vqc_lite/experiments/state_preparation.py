from tqdm import tqdm
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from ..circuit_stack.gate import Haar_Random
from ..toolboxes.toolbox_statevector import fidelity


class Compression_Sweeping:
    """
    The goal of compression is to iteratively optimize the parameters of the VQC, such that it outputs a statevector
    with high fidelity compared to the target state.

    This task uses a sweeping optimization method, inspired by some tensor network techniques.
    Ref. Real- and Imaginary-Time Evolution with Compressed Quantum Circuits. PRX Quantum 2, 010342

    :param psi_target: The target state to prepare
    :type psi_target: (jax) numpy array in tensor shape
    :param circuit: the VQC to be trained for state preparation
    :type circuit: Circuit object from circuit_stack. Usage with circuit_stack_Pennylane is not supported
    :param steps: the maximal number of optimization iterations, 1e3 by default
    :type steps: int
    :param ea: the maximal absolute error before stopping the optimization, 1e-10 by default
    :type ea: float
    :param er: the maximal relative error before stopping the optimization, 1e-10 by default
    :type er: float
    :param seed: the random seed for the initialization of parameters, 1 by default
    :type seed: jax random seed
    :param init_filled_circuit: initial values for the parameterized gates, None by default and random initialization
           will be triggered.
    :type init_filled_circuit: list of (jax) numpy arrays or None
    """

    def __init__(self, psi_target, circuit, **kwargs):
        self.psi_target = psi_target
        self.circuit = circuit

        self.optimizer = "Sweeping"
        self.steps = kwargs.get("steps", 1e3)  # The maximal iteration number,
        # to be overwritten later by the actual iteration number
        self.ea = kwargs.get("ea", 1e-10)  # The absolute convergence error
        self.er = kwargs.get("er", 1e-10)  # The relative convergence error
        self.seed = kwargs.get("seed", 1)  # Random seed

        self.circuit.filled_circuit = kwargs.get("init_filled_circuit", None)
        if self.circuit.filled_circuit is None:
            self.__initialize_filled_circuit()

        self.record = None

    def __repr__(self):
        return "Compression_Sweeping({},{})".format(self.psi_target, self.circuit)

    def __str__(self):
        return "Compression_Sweeping_" + str(self.circuit)

    def __initialize_filled_circuit(self):
        """
        Initialize a filled_circuit with Haar random unitaries.
        """
        key = jax.random.PRNGKey(self.seed)
        self.circuit.fill_params(np.ones(self.circuit.get_np()))  # just to create a trivial filled_circuit first
        for ig in self.circuit.get_where_parametrized():
            global_index = self.circuit.il[ig]
            nq = len(global_index)
            key, subkey = jax.random.split(key)
            U = Haar_Random(nq, subkey).reshape(*tuple(2 * len(global_index) * [2]))
            self.circuit.filled_circuit[ig] = U

    def run(self):
        """
        Call this function to start the task.
        """
        self.record = {'optimized_gates': [], 'fidelity': []}

        psi_original = self.psi_target
        psi_in = np.zeros(self.circuit.nq * [2])
        psi_in[tuple(self.circuit.nq * [0])] = 1

        step = 0
        absolute_error = 1
        relative_error = 1

        while step < self.steps and absolute_error > self.ea and relative_error > self.er:
            step += 1

            for ig in self.circuit.get_where_parametrized():

                global_index = self.circuit.il[ig]

                unitary_shape = tuple(2 * [2 ** len(global_index)])
                tensor_shape = tuple(2 * len(global_index) * [2])

                E = self.circuit.environment_contraction(psi_in, psi_original.conjugate(),
                                                         ig).reshape(*unitary_shape)
                X, Sigma, Ydagger = jnp.linalg.svd(E)
                U = jnp.transpose(jnp.conjugate(X @ Ydagger))

                self.circuit.filled_circuit[ig] = U.reshape(*tensor_shape)

            psi_out = self.circuit.forward_contraction(psi_in)
            f = fidelity(psi_out, psi_original)
            self.record['fidelity'].append(f)

            pre_error = absolute_error
            absolute_error = 1 - f  # Error is defined by 1 - fidelity
            relative_error = np.abs(absolute_error - pre_error) / np.abs(pre_error)

        self.record['optimized_gates'] = [self.circuit.filled_circuit[ig]
                                          for ig in self.circuit.get_where_parametrized()]
        # only store parametrized gates

        self.steps = step  # replace the maximum step by the actual used step


class Compression_Adam:
    """
    The goal of compression is to iteratively optimize the parameters of the VQC, such that it outputs a statevector
    with high fidelity compared to the target state.

    This task uses Adam optimizer, a popular gradient based optimization method for machine learning and optimization.

    :param psi_target: The target state to prepare
    :type psi_target: (jax) numpy array in tensor shape
    :param circuit: the VQC to be trained for state preparation
    :type circuit: Circuit object from circuit_stack or Circuit_P object from circuit_stack_pennylane (recommended)
    :param steps: the maximal number of optimization iterations, 1e3 by default
    :type steps: int
    :param lr: the learning rate
    :type lr: float
    :param lf: the loss function. See circuit.py from circuit_stack or circuit_stack_pennylane
    :type lf: float
    :param seed: the random seed for the initialization of parameters, 1 by default
    :type seed: jax random seed
    :param init_params: initial values for the parameterized gates, None by default and random initialization
           will be triggered.
    :type init_params: (jax) numpy array
    """

    def __init__(self, psi_target, circuit, **kwargs):
        self.psi_target = psi_target
        self.circuit = circuit

        self.optimizer = "Adam"
        self.lr = kwargs.get("lr", 0.03)  # The learning rate
        self.optimizer_function = optax.adam(self.lr)  # By default, use Adam optimizer
        self.steps = kwargs.get('steps', 5000)  # The number of iterations
        self.seed = kwargs.get("seed", 1)  # Random seed
        self.lf = kwargs.get("lf", "1-f")  # Loss function

        self.init_params = kwargs.get("init_params", None)
        if self.init_params is None:
            self.initialize_params()

        self.record = None

    def __repr__(self):
        return "Compression_Adam({},{})".format(self.psi_target, self.circuit)

    def __str__(self):
        return "Compression_Adam_" + str(self.circuit)

    def initialize_params(self):
        """
        Initialize a filled_circuit with Haar random unitaries.
        """
        key = jax.random.PRNGKey(self.seed)
        self.init_params = jax.random.uniform(key, shape=tuple([self.circuit.get_np()]))
        if hasattr(self.circuit, "param_shape"):  # For Pennylane circuits, we need to reshape the parameters
            self.init_params = self.circuit.params_to_proper_shape(self.init_params)

    def run(self):
        """
        Call this function to start the task.
        """
        self.record = {'params': [], 'loss': [], 'fidelity': [], 'time': 0}

        def optimization_step(params, state, psi_original):
            """
            The iterative step in the optimization
            :param params: The parameters of the gates
            :param state: current state
            :param psi_original: the reference state, FRQI state
            :return:
            """
            grads = jax.grad(self.circuit.get_loss)(params, psi_original, lf=self.lf)
            # computes the gradient of the loss function at the current params
            updates, state = self.optimizer_function.update(grads, state)
            # opt.update outputs the computed gradient updates and the new state
            params = optax.apply_updates(params, updates)
            return params, state

        params = self.init_params
        state = self.optimizer_function.init(self.init_params)
        psi_original = self.psi_target

        start_time = time.monotonic()
        for _ in tqdm(range(self.steps)):
            params, state = optimization_step(params, state, psi_original)
            self.record['loss'].append(self.circuit.get_distance_with(params, psi_original))  # compute loss (distance)
            self.record['fidelity'].append(self.circuit.get_fidelity_with(params, psi_original))  # compute overlap
        end_time = time.monotonic()

        self.record['time'] = round((end_time - start_time) / self.steps, 3)

        self.record['params'] = np.concatenate([np.ravel(component) for component in params])  # flatten
