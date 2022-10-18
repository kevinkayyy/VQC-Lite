import pennylane as qml
from ..circuit_stack.gate import GU1, GU2


class Block_P:
    """
    The Pennylane version of the circuit stack consists of 3 levels, which from bottom to top are: gate, block and
    circuit.

    Note that because of functions such as "qml.MPS" and "qml.broadcast", the arrangement of blocks could be
    easily handled at the circuit level for many common Ansätze. Thus, at the moment the "layer" level which exists in
    the other circuit stack is omitted here. However, when more complicated Ansätze come in the future, maybe it's
    better to also introduce the layer here.

    In the other stack where attributes are passed from one level to the next and only assembled and mapped to gate
    functions at the circuit level. In the Pennylane stack, in contrary, we directly pass callable gate functions called
    "circuit".

    A Block is the repeating unit in a layer. A block consists of >= 1 parmaterized or fixed gate(s).

    :param pm1: parametrization method for GU1
    :type pm1: int
    """
    def __init__(self, pm1=0):
        self.circuit = None
        self.np = 0
        self.assemble()
        self.pm1 = pm1

    def __repr__(self):
        return "Block_P({})".format(self.pm1)

    def __str__(self):
        return "B_"

    def assemble(self):
        """
        Automatically called during initialization to generate the circuit function.
        """
        pass


class Block_GU1_P(Block_P):
    """
    This block consists of only 1 general single qubit unitary. It's takes 3 parameters.
    """
    def __init__(self, pm1=0):
        super(Block_GU1_P, self).__init__(pm1=pm1)
        self.np = 3

    def __repr__(self):
        return "Block_GU1_P({})".format(self.pm1)

    def __str__(self):
        return "B-G1_"

    def assemble(self):
        def circuit(weights, wires):
            qml.QubitUnitary(GU1(weights, parametrization=self.pm1), wires=wires)
        self.circuit = circuit


########################################################################################################################

class Block_GU2_P(Block_P):
    """
    This block consists of only 1 general two qubit unitary (GU2). It takes 15 parameters.
    """
    def __init__(self):
        super(Block_GU2_P, self).__init__()
        self.np = 15
        self.pm1 = 'N/A'

    def __repr__(self):
        return "Block_G2QU_P()"

    def __str__(self):
        return "B-G2_"

    def assemble(self):
        def circuit(weights, wires):
            qml.QubitUnitary(GU2(weights), wires=wires)
        self.circuit = circuit


class Block_CZ_P(Block_P):
    """
    This block consists of a Controlled-Z gate, followed by 2 general single qubit unitaries, one on each qubit. It
    takes 6 parameters.
    """
    def __init__(self, pm1=0):
        super(Block_CZ_P, self).__init__(pm1=pm1)
        self.np = 6

    def __repr__(self):
        return "Block_CZ_P({})".format(self.pm1)

    def __str__(self):
        return "B-CZ_"

    def assemble(self):
        def circuit(weights, wires):
            qml.CZ(wires=wires)
            qml.QubitUnitary(GU1(weights[0:3], parametrization=self.pm1), wires=wires[0])
            qml.QubitUnitary(GU1(weights[3:6], parametrization=self.pm1), wires=wires[1])
        self.circuit = circuit


class Block_CNOT_P(Block_P):
    """
    This block consists of a Controlled-X gate, followed by 2 general single qubit unitaries (GU1), one on each qubit.
    It takes 6 parameters.
    """
    def __init__(self, pm1=0):
        super(Block_CNOT_P, self).__init__(pm1=pm1)
        self.np = 6

    def __repr__(self):
        return "Block_CNOT_P({})".format(self.pm1)

    def __str__(self):
        return "B-CNOT_"

    def assemble(self):
        def circuit(weights, wires):
            qml.CNOT(wires=wires)
            qml.QubitUnitary(GU1(weights[0:3], parametrization=self.pm1), wires=wires[0])
            qml.QubitUnitary(GU1(weights[3:6], parametrization=self.pm1), wires=wires[1])
        self.circuit = circuit
