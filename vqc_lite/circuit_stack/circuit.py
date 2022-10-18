import numpy as np
import copy
from .gate import *
from .layer import Layer
from ..toolboxes.toolbox_statevector import distance, fidelity


class Circuit:
    """
    The circuit stack consists of 4 levels, which from bottom to top are: gate, block, layer and circuit.

    Circuits are at the top level of the stack.

    1. The most important attributes of a circuit are also "gs", "il" and "npl",
       which are formed by concatenating the ones of its layer components.
    2. The attribute "components" is a list of layers,
       which could be of different kinds.
    3. The attribute "filled_circuit" is a list of arrays,
       each corresponds to a fixed gate, or a parametrized gate filled with parameters.

    :param nq: the number of qubits in the circuit.
    :type nq: int
    :param pmd: the parameterization method dictionary, which defines how a gate should be parametrized if there are
                multiple ways to do so
    :type pmd: dictionary mapping str to int
    """
    def __init__(self, nq=4, **kwargs):
        self.nq = nq
        self.components = []  # layers
        self.gs, self.il, self.npl = [], [], []
        self.filled_circuit = None
        self.pmd = kwargs.get('pmd', {'GU1': None})

    def __repr__(self):
        return "Circuit({})".format(self.nq)

    def __str__(self):
        return "C-{}Q_".format(self.nq)

    def __add__(self, other):
        """
        We distinguish two cases for circuit addition.

        1. When we add with another circuit, a new circuit will be generated,
           the components of the two circuits will be added together, and the "pmd" of the circuit will be given by the
           other circuit.
        2. When we add the circuit with a layer, we will keep the circuit,
           the layer will simply be appended to the components. Then we will reassemble the circuit.

        :param other: circuit or layer to be added
        :type other: Circuit or Layer
        :return: new or modified circuit
        :rtype: Circuit
        """
        if isinstance(other, Circuit):
            new_circuit = copy.deepcopy(other)
            new_circuit.components = self.components + other.components
            new_circuit.assemble()
            return new_circuit
        if isinstance(other, Layer):
            self.components.append(other)
            self.assemble()
            return self

    def get_np(self):
        """
        Compute the total number of parameters in the block.

        :rtype: int
        """
        return sum(self.npl)

    def get_where_parametrized(self):
        """
        Return an array of integer indices, telling which gates of the gate sequence are parametrized gates

        :rtype: array of int
        """
        return np.nonzero(self.npl)[-1]

########################################################################################################################
    def assemble(self):
        """
        Automatically called during initialization to generate the three attributes.
        """
        self.npl = []
        self.gs = []
        self.il = []
        for component in self.components:
            self.npl += component.npl
            self.gs += component.gs
            self.il += component.il

########################################################################################################################
    def run_with_param_input(self, params):
        """
        Run the full circuit with parameter inputs for the parametrized gates.

        First compute the filled circuit and then perform a full forward contraction starting from state |0>.

        :param params: parameter inputs
        :type params: (jax) numpy array / list
        """
        psi = np.zeros(self.nq * [2])
        psi[tuple(self.nq * [0])] = 1

        self.fill_params(params)

        return self.forward_contraction(psi)

    def run_with_gate_input(self, gates):
        """
        Run the full circuit with gate inputs that would replace the parametrized gates and form the filled circuit
        together with the fixed gates in the gate sequence.

        First compute the filled circuit and then perform a full forward contraction starting from state |0>.

        :param gates: gate inputs
        :type gates: list of (jax) numpy arrays
        """
        psi = np.zeros(self.nq * [2])
        psi[tuple(self.nq * [0])] = 1

        self.fill_params(np.ones(self.get_np()))  # first create a trivial filled_circuit first
        for i in range(len(gates)):
            ig = self.get_where_parametrized()[i]  # find where the parametrized gates are
            self.filled_circuit[ig] = gates[i]  # replace the parametrized gates

        return self.forward_contraction(psi)

    def fill_params(self, params):
        """
        Form the filled_circuit by filling parameters into the parametrized gates.

        :param params: parameter inputs
        :type params: (jax) numpy array / list
        """
        param_count = 0
        gate_count = 0
        filled_circuit = []

        while gate_count < len(self.gs):

            gate_func = self.gs[gate_count]
            shape = tuple([2] * 2 ** len(self.il[gate_count]))  # compute the shape of the gate
            nparam = self.npl[gate_count]

            param = params[param_count: param_count + nparam]
            param_count += nparam

            if nparam > 0:  # not empty, parametrized gate
                if gate_func in self.pmd.keys():  # gate that has different parametrization methods
                    gate = globals()[gate_func](param, parametrization=self.pmd[gate_func])
                else:  # gate that has only one parameterization method
                    gate = globals()[gate_func](param).reshape(*shape)
            else:  # empty, fixed gate
                gate = globals()[gate_func]().reshape(*shape)

            filled_circuit.append(gate)

            gate_count += 1

        self.filled_circuit = filled_circuit

    def forward_contraction(self, psi_in, ng=np.inf):
        """
        Contract the filled circuit in forward direction until some stopping point, and compute the output state

        :param psi_in: A ket state. Usually |0>.
        :type psi_in: (jax) numpy array
        :param ng: The number of gates to contract before stop.
        :type ng: int

        :return: ket state after contraction
        :rtype: (jax) numpy array
        """
        leg_list = [[1], [2, 3]]  # for storing the legs of the gate during contraction

        def contraction(circuit, il, nq):
            psi = psi_in

            for gate_count in range(min(ng, len(circuit))):
                gate = circuit[gate_count]
                global_index = il[gate_count]

                n = len(global_index)

                psi = jnp.tensordot(gate, psi, [leg_list[n-1], global_index])

                order = list(range(n, nq))
                for i in range(n):
                    order.insert(global_index[i] % nq, i)

                psi = jnp.transpose(psi, axes=tuple(order))

            return psi

        return contraction(self.filled_circuit, self.il, self.nq)

    def backward_contraction(self, psi_out, ng=np.inf):
        """
        Contract the filled circuit in backward direction until some stopping point and compute the input state.

        :param psi_out: A bra state. When using in optimization, if psi is the target state, psi_out = psi.conjugate()
        :type psi_out: (jax) numpy array
        :param ng: The number of gates to contract before stop.
        :type ng: int
        :return: ket state after contraction
        :rtype: (jax) numpy array
        """
        leg_list = [[0], [0, 1]]  # for storing the legs of the gate during contraction

        def contraction(circuit, il, nq):
            psi = psi_out

            for gate_count in range(min(ng, len(circuit))):
                gate = circuit[len(circuit) - gate_count - 1]
                global_index = il[len(circuit) - gate_count - 1]

                n = len(global_index)

                psi = jnp.tensordot(gate, psi, [leg_list[n - 1], global_index])

                order = list(range(n, nq))
                for i in range(n):
                    order.insert(global_index[i] % nq, i)

                psi = jnp.transpose(psi, axes=tuple(order))

            return psi

        return contraction(self.filled_circuit, self.il, self.nq)

    def environment_contraction(self, psi_in, psi_out, ig):
        """
        Compute the environment tensor of a gate, by performing forward contraction from the input state, and backward
        contraction from the output state, toward the gate. This function is usually called during sweeping
        optimization.

        :param psi_in: A ket state. Usually |0>.
        :type psi_in: (jax) numpy array
        :param psi_out: A bra state. When using in optimization, if psi is the target state, psi_out = psi.conjugate()
        :type psi_out: (jax) numpy array
        :param ig: The index of the target gate. We start counting from 1!
        :type ig: int
        :return: The environment tensor
        :rtype: (jax) numpy array
        """
        psi_forward = self.forward_contraction(psi_in, ng=ig)  # perform forward contraction for ig gates
        psi_backward = self.backward_contraction(psi_out, ng=len(self.filled_circuit) - ig - 1)
        # perform backward contraction for len(circuit) - ig - 1 gates
        global_index = self.il[ig]  # The qubits the gate acts on.
        contract_index = [i for i in list(range(self.nq)) if i not in global_index]  # The indices to be contracted,
        # basically all the qubits that the gate does not act on.
        environment_tensor = jnp.tensordot(psi_forward, psi_backward, [contract_index, contract_index])

        return environment_tensor

########################################################################################################################
    def get_loss(self, params, psi, lf='td'):
        """
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
