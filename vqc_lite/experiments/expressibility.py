import numpy as np
import matplotlib.pyplot as plt
import jax


class Expressibility_Evaluation:
    """
    Expressibility is a popular descriptor of VQCs. It compares the output statevectors of a VQC with Haar random states
    of the corresponding Hilbert space. It is computed as the KL-divergence between two histograms. For more details,
    please refer to:

    Reference: Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical
    algorithms. S. Sim et al. arXiv: 1905.10876

    :param circuit: the VQC to be evaluated
    :type circuit: Circuit object from circuit_stack or Circuit_P object from circuit_stack_pennylane (recommended)
    :param n_samples: the number of pairs of statevectors to sample, 10000 by default
    :type n_samples: int
    :param n_bins: the number of bins for the histogram, 75 by default
    :type n_bins: int
    :param seed: the random seed for the initialization of parameters, 1 by default
    :type seed: jax random seed
    :param parameter_bounds: the interval in which parameters are sampled, [0, 1] by default
    :type parameter_bounds: [float, float]
    """
    def __init__(self, circuit, **kwargs):
        self.circuit = circuit
        self.n_samples = kwargs.get('n_samples', 10000)
        self.n_bins = kwargs.get('n_bins', 75)
        self.seed = kwargs.get("seed", 1)
        self.parameter_bounds = kwargs.get("parameter_bounds", [0, 1])
        self.parameter_samples = None
        self.fidelity_samples = None
        self.bin_edges = None
        self.bin_heights = None
        self.bin_heights_harr = None
        self.DKL = None

    def __repr__(self):
        return "Expressibility_Evaluation({})".format(self.circuit)

    def __str__(self):
        return "Expressibility_Evaluation" + str(self.circuit)

    def __get_parameter_samples(self):
        np = self.circuit.get_np()
        key = jax.random.PRNGKey(self.seed)
        parameter_samples = jax.random.uniform(key, shape=tuple([np * 2 * self.n_samples]),
                                               minval=self.parameter_bounds[0], maxval=self.parameter_bounds[1])
        # First get the parameters in an array, and then divide them into 2 * n_sample portions.
        parameter_samples = [parameter_samples[i: i + np] for i in range(0, len(parameter_samples), np)]
        if hasattr(self.circuit, "param_shape"):  # For Pennylane circuits, we need to reshape the parameters
            parameter_samples = [self.circuit.params_to_proper_shape(params) for params in parameter_samples]
        self.parameter_samples = parameter_samples

    def __get_fidelity_samples(self):
        fidelity_samples = []
        for i in range(self.n_samples):
            psi0 = self.circuit.run_with_param_input(self.parameter_samples[2 * i])  # statevector of the first sample
            fidelity = self.circuit.get_fidelity_with(self.parameter_samples[2 * i + 1], psi0)
            # compare with statevector of the second sample
            fidelity_samples.append(np.array(fidelity))
        self.fidelity_samples = np.array(fidelity_samples)
        self.fidelity_samples.reshape(-1)  # flatten to an array

    def __get_histogram_counts(self):
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        weights = np.ones_like(self.fidelity_samples) / len(self.fidelity_samples)
        self.bin_heights, self.bin_edges = np.histogram(self.fidelity_samples, bins=bin_edges, weights=weights)

    def __compute_Harr(self):
        N = 2 ** self.circuit.nq
        n_bins = len(self.bin_edges) - 1
        self.bin_heights_harr = np.clip(np.array([(1 - self.bin_edges[i]) ** (N - 1) -
                                                  (1 - self.bin_edges[i + 1]) ** (N - 1)
                                                  for i in range(n_bins)]), 1e-300, 1)

    def __compute_DKL(self):
        self.DKL = np.sum(self.bin_heights * np.log(np.clip(self.bin_heights / self.bin_heights_harr, 1e-300, 1e300)))

    def __plot(self):
        fig, ax = plt.subplots(1, 1)
        bin_centers = (self.bin_edges + self.bin_edges[1] / 2)[:-1]
        width = 2 * bin_centers[0]
        ax.bar(bin_centers, self.bin_heights, width=width, label=str(self.circuit), color="#226E9C", linewidth=0)
        ax.plot(bin_centers, self.bin_heights_harr, label="Harr", color="#0D4A70")
        ax.legend(loc="upper right")
        ax.text(0.4, 0.95, "DKL = " + str(round(self.DKL, 3)), ha="right", va="center", transform=ax.transAxes)
        ax.set_xlabel("Fidelity")
        ax.set_ylabel("Frequency")
        ax.set_title("Fidelity between states randomly sampled from parameter space, " + str(self.n_samples) + " pairs")
        plt.show()

    def run(self):
        """
        Call this function to start the task.
        """
        self.__get_parameter_samples()
        self.__get_fidelity_samples()
        self.__get_histogram_counts()
        self.__compute_Harr()
        self.__compute_DKL()
        self.__plot()


