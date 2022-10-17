# VQC-Lite

VQC-Lite is an object-oriented Python library for building variational quantum circuits and related tasks in quantum 
machine learning (QML). 

VQC-Lite may not be the most efficient simulator, but it's definitely an intuitive, reader-friendly and simple-to-use one. With 
just a few hundred lines of code and full documentation, one could see how a VQC is built layer by layer, from 
elementary unitary gates all the way up to a differentiable and optimizable quantum circuit. 

VQC-Lite provides two alternative ways to construct VQCs, one involving only standard numpy functions, and one using the 
framework Pennylane (https://github.com/PennyLaneAI/pennylane). Both implementations give VQCs of similar 
functionalities with the same I/O. While the latter is currently more efficiently integrated with Jax for circuit 
optimization by differentiation, the former provides the additional feature of optimization by sweeping.

Author and Developer: Kevin Shen (www.linkedin.com/in/kevinshen-tum)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install VQC-Lite.

```bash
pip install VQC-Lite
```

Use the requirements.txt sheet to install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Please read the Jupyter Notebooks under the folder demo/ for more details. There contains 2 notebooks.

The notebook "circuit" gives an introduction to the VQC circuit stacks.

The notebook "experiment" gives an illustrative example application for VQC: uploading an image onto a quantum computer.

```python
import numpy as np
from circuit_stack.circuit_mps import MPS_GU2
from experiments.expressibility import Expressibility_Evaluation
from experiments.state_preparation import Compression_Sweeping

# initialize a VQC
circuit = MPS_GU2(nl=1, nq=4)  # 1 layer of general 2 qubit gates covering 4 qubits

# execute a VQC
params = np.random.uniform(size=45)  # 3 gates, each with 15 parameters
psi_out = circuit.run_with_param_input(params)  # get output state

# state preparation with VQC
ghz = np.zeros(4 * [2])  # Let's compress a GHZ state as an example
ghz[0,0,0,0] = 1 / np.sqrt(2)
ghz[1,1,1,1] = 1 / np.sqrt(2)
task = Compression_Sweeping(ghz, circuit)
task.run()

# evaluate expressibility
task = Expressibility_Evaluation(circuit)
task.run()

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)