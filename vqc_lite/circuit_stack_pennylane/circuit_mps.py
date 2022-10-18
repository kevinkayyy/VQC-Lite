import numpy as np
from .circuit import Circuit_P
from .block import *


class MPS_P(Circuit_P):
    """
    This is the basis class for all circuits inspired by matrix product states (MPS).

    :param block: The block object underlying the circuit.
    :type block: Block_P
    :param nl: The number of layers in the circuit.
    :type nl: int
    :param nq: The number of qubits in the circuit.
    :type nq: int
    :param irb: The block object for initial rotation, if applicable. It's None by default.
    :type irb: Block
    :param version: define the block arrangement in:

                 0. right canonical form
                 1. right canonical form with periodic condition
                 2. mixed canonical form

    :type version: int
    """
    def __init__(self, block, nl=1, nq=11, irb=None, version=0, **kwargs):
        super(MPS_P, self).__init__(nq, **kwargs)
        self.block = block
        self.nl = nl
        self.nb = self.nq - 1
        self.irb = irb
        self.version = version

        def initial_rotation(params):
            for q in range(self.nq):
                self.irb.circuit(params[q], q)

        if version == 1:  # right canonical form with periodic boundary condition
            def ladder_layers(params):
                for layer in range(self.nl):
                    for i in range(self.nq):
                        p = params[layer][i]
                        self.block.circuit(p, (i, (i+1) % self.nq))

            self.param_shape = [(self.nl, self.nq, self.block.np)]

        elif version == 2:  # mixed canonical form
            def ladder_layers(params):
                for layer in range(self.nl):
                    mi = int(np.ceil(self.nq / 2)) - 1  # middle index
                    for i in range(mi):
                        pr = params[layer][2*i]
                        self.block.circuit(pr, (mi + i, mi + i + 1))
                        pl = params[layer][2*i + 1]
                        self.block.circuit(pl, (mi - 1 - i, mi - i))
                    if self.nq % 2 == 0:
                        pr = params[layer][-1]
                        self.block.circuit(pr, (self.nq - 2, self.nq - 1))

            self.param_shape = [(self.nl, self.nb, self.block.np)]

        else:
            def ladder_layers(params):
                for layer in range(self.nl):
                    qml.MPS(
                        wires=range(self.nq),
                        n_block_wires=2,
                        block=self.block.circuit,
                        n_params_block=self.block.np,
                        template_weights=params[layer],
                    )

            self.param_shape = [(self.nl, self.nb, self.block.np)]

        if self.irb is not None:
            self.param_shape = [(self.nq, 3)] + self.param_shape
            self.components = [initial_rotation, ladder_layers]
        else:
            self.components = [ladder_layers]

        self.assemble()

    def __repr__(self):
        return "MPS_P({}, {}, {}, {}, {})".format(repr(self.block), self.nl, self.nq, repr(self.irb), self.version)

    def __str__(self):
        return "PC-{}L{}Q_".format(self.nl, self.nq) + str(self.block)


class MPS_GU2_P(MPS_P):
    """
    MPS with GU2 blocks.
    """
    def __init__(self, nl=1, nq=11, version=0, **kwargs):
        block = Block_GU2_P()
        super(MPS_GU2_P, self).__init__(block, nl, nq, irb=None, version=version, **kwargs)

    def __repr__(self):
        return "MPS_GU2_P({}, {}, {})".format(self.nl, self.nq, self.version)


class MPS_CZ_P(MPS_P):
    """
    MPS with CZ blocks.

    :param pm1: parameterization method for GU1. See gate.py
    :type pm1: int
    """
    def __init__(self, nl=1, nq=11, irb=None, version=0, pm1=0, **kwargs):
        self.pm1 = pm1
        block = Block_CZ_P(pm1=self.pm1)
        super(MPS_CZ_P, self).__init__(block, nl, nq, irb, version=version, **kwargs)

    def __repr__(self):
        return "MPS_CZ_P({}, {}, {}, {}, {})".format(self.nl, self.nq, repr(self.irb), self.version, self.pm1)


class MPS_CNOT_P(MPS_P):
    """
    MPS with CNOT blocks.

    :param pm1: parameterization method for GU1. See gate.py
    :type pm1: int
    """
    def __init__(self, nl=1, nq=11, irb=None, version=0, pm1=0, **kwargs):
        self.pm1 = pm1
        block = Block_CNOT_P(pm1=self.pm1)
        super(MPS_CNOT_P, self).__init__(block, nl, nq, irb, version=version, **kwargs)

    def __repr__(self):
        return "MPS_CNOT_P({}, {}, {}, {}, {})".format(self.nl, self.nq, repr(self.irb), self.version, self.pm1)
