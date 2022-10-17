class Block:
    """
    The circuit stack consists of 4 levels, which from bottom to top are: gate, block, layer and circuit.

    A block is the repeating unit in a layer. A block consists of >= 1 parameterized or fixed gate(s).
    The most important attributes of a block are three lists, called "gs", "il" and "npl".

    1. "gs" stands for gate sequence,
        which is a list of strings (names of the gates). The functions realizing the gates
        are defined in the file gate.py. The strings will be mapped to the functions at the circuit level.
    2. "il" stands for index list,
        which is a list of lists of integers called 'local indices', which defines the qubits on which
        the gates apply.
    3. "npl" stands for number of parameter list,
        which is a list storing the number of parameters each gate takes.
    """
    def __init__(self):
        self.gs, self.il, self.npl = [], [], []
        # npl stands for "number-of-parameters list", which stores the number of parameters each gate in the block takes
        # gs stands for "gate_sequence", which stores the name of each gate in the block, which will eventually be used
        # to call the gate functions at the circuit level.
        # il stands for "index_list",
        self.assemble()

    def __repr__(self):
        return "Block()"

    def __str__(self):
        return "B_"

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


class Block_GU1(Block):
    """
    This block consists of only 1 general single qubit unitary. It's takes 3 parameters.
    """
    def __init__(self):
        super(Block_GU1, self).__init__()

    def __repr__(self):
        return "Block_GU1()"

    def __str__(self):
        return "B-GU1_"

    def assemble(self):
        self.gs = ['GU1']
        self.il = [[0]]
        self.npl = [3]


class Block_GU2(Block):
    """
    This block consists of only 1 general two qubit unitary (GU2). It takes 15 parameters.
    """
    def __init__(self):
        super(Block_GU2, self).__init__()

    def __repr__(self):
        return "Block_GU2()"

    def __str__(self):
        return "B-GU2_"

    def assemble(self):
        self.gs = ['GU2']
        self.il = [[0, 1]]
        self.npl = [15]


class Block_CZ(Block):
    """
    This block consists of a Controlled-Z gate, followed by 2 general single qubit unitaries, one on each qubit. It
    takes 6 parameters.
    """
    def __init__(self):
        super(Block_CZ, self).__init__()

    def __repr__(self):
        return "Block_CZ()"

    def __str__(self):
        return "B-CZ_"

    def assemble(self):
        self.gs = ['CZ', 'GU1', 'GU1']
        self.il = [[0, 1], [0], [1]]
        self.npl = [0, 3, 3]


class Block_CNOT(Block):
    """
    This block consists of a Controlled-X gate, followed by 2 general single qubit unitaries (GU1), one on each qubit.
    It takes 6 parameters. As an example, the three attributes for this block writes:

    1. gs = ['CNOT', 'GU1', 'GU1']
    2. il = [[0, 1], [0], [1]]
    3. npl = [0, 3, 3]
    """
    def __init__(self):
        super(Block_CNOT, self).__init__()

    def __repr__(self):
        return "Block_CNOT()"

    def __str__(self):
        return "B-CNOT_"

    def assemble(self):
        self.gs = ['CNOT', 'GU1', 'GU1']
        self.il = [[0, 1], [0], [1]]
        self.npl = [0, 3, 3]