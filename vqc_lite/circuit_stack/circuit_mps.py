from .circuit import Circuit
from .layer import *
from .block import *


class MPS(Circuit):
    """
    This is the basis class for all circuits inspired by matrix product states (MPS).

    :param layer: The layer object underlying the circuit.
    :type layer: Layer
    :param nl: The number of layers in the circuit.
    :type nl: int
    :param nq: The number of qubits in the circuit.
    :type nq: int
    :param irb: The block object for initial rotation, if applicable. It's None by default.
    :type irb: Block
    """
    def __init__(self, layer, nl=1, nq=4, irb=None, **kwargs):
        super(MPS, self).__init__(nq, **kwargs)
        self.layer = layer
        self.nb = self.nq - 1
        self.nl = nl
        self.irb = irb
        self.components = [self.layer] * nl

        if self.irb is not None:
            self.components = [IRLayer(self.irb)] + self.components

        self.assemble()

    def __repr__(self):
        return "MPS({}, {}, {}, {})".format(repr(self.layer), self.nl, self.nq, repr(self.irb))

    def __str__(self):
        return "C-{}L{}Q_".format(self.nl, self.nq) + str(self.layer.block)


class MPS_GU2(MPS):
    """
    MPS with GU2 blocks.

    :param version: define the block arrangement in:

                    0. right canonical form
                    1. right canonical form with periodic condition
                    2. mixed canonical form

    :type version: int

    """
    def __init__(self, nl=1, nq=4, version=0, **kwargs):
        if version == 1:
            layer = MCMPSLayer(Block_GU2(), nq=nq)
        elif version == 2:
            layer = PRCMPSLayer(Block_GU2(), nq=nq)
        else:
            layer = RCMPSLayer(Block_GU2(), nq=nq)
        self.version = version
        super(MPS_GU2, self).__init__(layer, nl, nq, **kwargs)

    def __repr__(self):
        return "MPS_GU2({}, {}, {})".format(self.nl, self.nq, self.version)


class MPS_CZ(MPS):
    """
    MPS with CZ blocks.

    :param pm1: parameterization method for GU1. See gate.py
    :type pm1: int
    :param version: define the block arrangement in:

                    0. right canonical form
                    1. right canonical form with periodic condition
                    2. mixed canonical form

    :type version: int

    """
    def __init__(self, nl=1, nq=4, irb=None, pm1=0, version=0, **kwargs):
        if version == 1:
            layer = MCMPSLayer(Block_CZ(), nq=nq)
        elif version == 2:
            layer = PRCMPSLayer(Block_CZ(), nq=nq)
        else:
            layer = RCMPSLayer(Block_CZ(), nq=nq)
        self.version = version
        super(MPS_CZ, self).__init__(layer, nl, nq, irb, **kwargs)
        self.pmd['GU1'] = pm1

    def __repr__(self):
        return "MPS_CZ({}, {}, {}, {}, {})".format(self.nl, self.nq, repr(self.irb), self.pmd['GU1'], self.version)


class MPS_CNOT(MPS):
    """
    MPS with CNOT blocks.

    :param pm1: parameterization method for GU1. See gate.py
    :type pm1: int
    :param version: define the block arrangement in:

                    0. right canonical form
                    1. right canonical form with periodic condition
                    2. mixed canonical form

    :type version: int
    """
    def __init__(self, nl=1, nq=4, irb=None, pm1=0, version=0, **kwargs):
        if version == 1:
            layer = MCMPSLayer(Block_CNOT(), nq=nq)
        elif version == 2:
            layer = PRCMPSLayer(Block_CNOT(), nq=nq)
        else:
            layer = RCMPSLayer(Block_CNOT(), nq=nq)
        self.version = version
        super(MPS_CNOT, self).__init__(layer, nl, nq, irb, **kwargs)
        self.pmd['GU1'] = pm1

    def __repr__(self):
        return "MPS_CNOT({}, {}, {}, {}, {})".format(self.nl, self.nq, repr(self.irb), self.pmd['GU1'], self.version)
