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

Author and Developer: Kevin Shen -> [Github](https://github.com/kevinkayyy) -> [LinkedIn](https://www.linkedin.com/in/kevinshen-tum)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install VQC-Lite.

```bash
pip install vqc_lite
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
from vqc_lite.circuit_stack_pennylane.circuit_mps import MPS_GU2_P
from vqc_lite.experiments.expressibility import Expressibility_Evaluation
from vqc_lite.experiments.state_preparation import Compression_Adam

# initialize a VQC
circuit = MPS_GU2_P(nl=1, nq=4)  # 1 layer of general 2 qubit gates covering 4 qubits

# execute a VQC
params = np.random.uniform(size=45)  # 3 gates, each with 15 parameters
params = circuit.params_to_proper_shape(params)  # reshape the parameters
psi_out = circuit.run_with_param_input(params)  # get output state

# state preparation with VQC
ghz = np.zeros(4 * [2])  # Let's compress a GHZ state as an example
ghz[0, 0, 0, 0] = 1 / np.sqrt(2)
ghz[1, 1, 1, 1] = 1 / np.sqrt(2)
task = Compression_Adam(ghz, circuit, steps=1000)  # use Adam optimizer
task.run()

# evaluate expressibility
task = Expressibility_Evaluation(circuit)
task.run()

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)