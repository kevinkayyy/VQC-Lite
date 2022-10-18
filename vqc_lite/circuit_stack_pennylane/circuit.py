import numpy as np
import matplotlib.pyplot as plt
import copy
import jax.numpy as jnp
from jax import jit
import pennylane as qml
from ..toolboxes.toolbox_statevector import distance, fidelity


class Circuit_P:
    """
    The Pennylane version of the circuit stack consists of 3 levels, which from bottom to top are: gate, block and
    circuit.

    Circuits are at the top level of the stack.

    Some important attributes of a circuit include:

    1. nq: The number of qubits involved in the circuit.
    2. components: A list of circuits of the constituent layer(s).
    3. param_shape: The shape of parameters for each component.
    4. mq: "measure_qubits". By default, it is False, meaning all qubits are measured and the exact output statevector
       will be returned. Else, it could be a list of the indices of the qubits to be measured. Then the probability
       distribution of bitstrings of the output qubits will be returned.

    :param nq: the number of qubits
    :type nq: int
    :param mq: the qubits to be measured
    :type mq: list of integers or False
    """
    def __init__(self, nq=4, **kwargs):
        self.nq = nq
        self.components = []
        self.param_shape = []
        self.mq = kwargs.get('mq', False)  # False means measure all qubits
        self.circuit, self.circuit_pre = None, None
        self.device = qml.device("default.qubit.jax", wires=self.nq)

    def __repr__(self):
        return "Circuit_P({})".format(self.nq)

    def __str__(self):
        return "PC-{}Q_".format(self.nq)

    def __add__(self, other):
        """
        The components of the two circuits will be added together, and the "mq" of the circuit will be given by the
        other circuit.

        :param other: circuit to be added
        :type other: Circuit_P
        :return: new circuit
        :rtype: Circuit_P
        """
        new_circuit = copy.deepcopy(other)
        new_circuit.param_shape = self.param_shape + other.param_shape
        new_circuit.components = self.components + other.components
        new_circuit.assemble()
        return new_circuit

    def get_np(self):
        """
        Compute the total number of parameters in the block.

        :rtype: int
        """
        return sum([np.prod(shape) for shape in self.param_shape])

    def params_to_proper_shape(self, flattened_params):
        """
        Reshape a flattened array of parameters into the form ready to be fed into the circuit.

        :param flattened_params: flattened parameters
        :type flattened_params: (jax) numpy array / list of float
        :return: parameters reshaped according to param_shape
        :rtype: list of (jax) numpy arrays
        """
        reshaped_params = []
        index = 0
        for shape in self.param_shape:
            length = np.prod(shape)
            reshaped_params.append(flattened_params[index: index + length].reshape(shape))
            index += length
        return reshaped_params

########################################################################################################################
    def assemble(self):
        """
        Automatically called during initialization to generate the circuit function.
        """
        self.__assemble_1()
        self.__assemble_2()

    def __assemble_1(self):
        """
        Automatically called during initialization to generate the 'circuit-pre' function, which outputs the statevector
        in linear array form. 'circuit-pre' is called, for example, in 'plot'.
        """
        @qml.qnode(self.device, diff_method="backprop", interface="jax")
        def circuit_pre(params):
            for i in range(len(self.components)):
                self.components[i](params[i])
            if self.mq:
                return qml.probs(wires=self.mq)
            return qml.state()

        self.circuit_pre = circuit_pre

    def __assemble_2(self):
        """
        Automatically called during initialization to generate the 'circuit' function, which executes the "circuit_pre"
        function but additionally reshape the output statevector into tensor form. In the case mq is False, as the
        output of "circuit-pre" is a bitstring distribution instead of array, we leave "circuit" to be the same as
        "circuit-pre".
        """
        @jit
        def circuit_psi(params):
            nq_out = len(self.mq) if self.mq else self.nq
            return self.circuit_pre(params).reshape(nq_out * [2])

        if self.mq:
            self.circuit = self.circuit_pre
        else:
            self.circuit = circuit_psi

########################################################################################################################
    def run_with_param_input(self, params):
        """
        Run the full circuit with parameter inputs for the parametrized gates.

        :param params: parameter inputs
        :type params: (jax) numpy array / list
        """
        return self.circuit(params)

    def plot(self, file_name=None, **kwargs):
        """
        Make a plot of the circuit and optionally save as a file.

        :param file_name: The name of the file that saves the plot
        :type file_name: str
        :param fig_size: size of the image
        :type fig_size: tuple of int
        :param dpi: quality of the image
        :type dpi: int
        """
        qml.drawer.use_style("black_white")
        params = kwargs.get("params", [np.zeros(param_shape) for param_shape in self.param_shape])
        fig, ax = qml.draw_mpl(self.circuit_pre, expansion_strategy="device")(params)
        fig.set_size_inches(kwargs.get("fig_size", (20, 10)))
        if file_name is not None:
            fig.savefig(file_name, dpi=kwargs.get("dpi", 100))
            plt.close()
        else:
            plt.show()

    def to_qasm(self, params, file_name):
        """
        Outputs the gate sequence with parameter inputs in OpenQASM format. May fail, depending on the type of gates
        involved.

        :param params: parameter inputs
        :type params: (jax) numpy array / list
        :param file_name: The name of the file that saves the plot
        :type file_name: str
        """
        self.circuit_pre(params)
        text = self.circuit_pre.qtape.to_openqasm()
        file = open(file_name, "w+")
        file.write(text)
        file.close()

########################################################################################################################
    def get_loss(self, params, psi, lf='td'):
        """
        Applicable when mq is False.

        Compute the "loss" of the output state of the VQC with parameter inputs compared to a target state psi.

        Often called during optimization for state preparation. Could also be used for more general purposes.

        :param params: parameter inputs
        :type params: (jax) numpy array / list
        :param psi: target state
        :type psi: (jax) numpy array
        :param lf: choice of loss function, currently supports:

                      1. coordinate-wise distance
                      2. squared coordinate-wise distance
                      3. 1 - fidelity
                      4. trace distance
        :type lf: int
        """
        if lf == 'd':  # Vector distance
            return self.get_distance_with(params, psi)
        if lf == 'd2':  # Vector distance squared
            return self.get_distance_with(params, psi)**2
        if lf == "1-f":  # 1 - fidelity
            return 1 - self.get_fidelity_with(params, psi)
        if lf == "td":  # Trace distance, related to fidelity for pure states
            return 2 * jnp.sqrt(1 - self.get_fidelity_with(params, psi))

    def get_distance_with(self, params, psi):
        """
        Compute the coordinate-wise distance between the output state of the VQC with parameter inputs and a target
        state psi.

        :param params: parameter inputs
        :type params: (jax) numpy array / list
        :param psi: target state
        :type psi: (jax) numpy array
        """
        psi_out = self.run_with_param_input(params)
        return distance(psi_out, psi)

    def get_fidelity_with(self, params, psi):
        """
        Compute the fidelity between the output state of the VQC with parameter inputs and a target state psi.

        :param params: parameter inputs
        :type params: (jax) numpy array / list
        :param psi: target state
        :type psi: (jax) numpy array
        """
        psi_out = self.run_with_param_input(params)
        return fidelity(psi_out, psi)

########################################################################################################################