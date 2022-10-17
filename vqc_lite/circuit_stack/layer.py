class Layer:
    """
    The circuit stack consists of 4 levels, which from bottom to top are: gate, block, layer and circuit.

    A layer is a component of the circuit, which could repeat itself a few times in the circuit. A layer
    is an arrangement of repeating blocks that covers all qubits.

    The most important attributes of a circuit are also "gs", "il" and "npl", which are formed by concatenating the ones
    of its block components.

    :param block: The repeating block in the layer
    :type block: Block
    :param nq: The number of qubits
    :type nq: int
    """
    def __init__(self, block, **kwargs):
        self.block = block
        self.nq = kwargs.get('nq', 4)  # The number of qubits.
        self.gs, self.il, self.npl = [], [], []
        self.assemble()

    def __repr__(self):
        return "Layer({})".format(self.block)

    def __str__(self):
        pass

    def assemble(self):
        """
        Automatically called during initialization to generate the three attributes.
        """
        pass

    def get_np(self):
        """
        Compute the total number of parameters in the block.
        """
        return sum(self.npl)


class RCMPSLayer(Layer):
    """
    A ladder-shaped layer that grows in one direction, inspired by MPS of right canonical form.
    """
    def __init__(self, block, **kwargs):
        super(RCMPSLayer, self).__init__(block, **kwargs)

    def __repr__(self):
        return "RCMPSLayer({})".format(self.block)

    def assemble(self):
        for global_index in range(self.nq-1):
            self.gs += self.block.gs
            for local_index in self.block.il:
                self.il.append([i + global_index for i in local_index])
            self.npl += self.block.npl


class PRCMPSLayer(Layer):
    """
    A ladder-shaped layer that grows in one direction, inspired by MPS of right canonical form with periodic boundary
    condition.
    """
    def __init__(self, block, **kwargs):
        super(PRCMPSLayer, self).__init__(block, **kwargs)

    def __repr__(self):
        return "PRCMPSLayer({})".format(self.block)

    def assemble(self):
        for global_index in range(self.nq):
            self.gs += self.block.gs
            for local_index in self.block.il:
                self.il.append([(i + global_index) % self.nq for i in local_index])
            self.npl += self.block.npl


class MCMPSLayer(Layer):
    """
    A ladder-shaped layer that grows in both directions, inspired by MPS of mixed canonical form.
    """
    def __init__(self, block, **kwargs):
        super(MCMPSLayer, self).__init__(block, **kwargs)

    def __repr__(self):
        return "MCMPSLayer({})".format(self.block)

    def assemble(self):
        pass


class IRLayer(Layer):
    """
    A flat layer of single qubit unitaries. Typically used for initial rotations on the qubits.
    """
    def __init__(self, block, **kwargs):
        super(IRLayer, self).__init__(block, **kwargs)

    def __repr__(self):
        return "IRLayer({})".format(self.block)

    def assemble(self):
        for global_index in range(self.nq):
            self.npl += self.block.npl
            self.gs += self.block.gs
            self.il.append(([global_index]))
