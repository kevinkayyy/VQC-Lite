import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from itertools import product

#  Pauli Basis #########################################################################################################

def Id1():
    """
    The identity matrix for the 1Q space
    """
    return jnp.eye(2)


def PauliX():
    """
    Pauli-X matrix / 1Q X-gate
    """
    return jnp.array([[0, 1], [1, 0]])


def PauliY():
    """
    Pauli-Y matrix / 1Q Y-gate
    """
    return jnp.array([[0, -1.j], [1.j, 0]])


def PauliZ():
    """
    Pauli-Z matrix / 1Q Z-gate
    """
    return jnp.array([[1, 0], [0, -1]])


def PauliBasis1():
    """
    1-Q Pauli basis stored in array
    """
    return jnp.array([Id1(), PauliX(), PauliY(), PauliZ()])


def PauliBasis2():
    """
    2-Q Pauli basis stored in array
    """
    return jnp.array([jnp.kron(pair[0], pair[1]) for pair in list(product(PauliBasis1(), PauliBasis1()))])


#  Parametrized 1Q gates ###############################################################################################

def RX(x):
    """
    The parametrized RX-gate

    :param x: the rotation angle
    :type x: float
    """
    return jnp.array([[jnp.cos(x / 2), -1j * jnp.sin(x / 2)], [-1j * jnp.sin(x / 2), jnp.cos(x / 2)]])


def RY(x):
    """
    The parametrized RY-gate

    :param x: the rotation angle
    :type x: float
    """
    return jnp.array([[jnp.cos(x / 2), -jnp.sin(x / 2)], [jnp.sin(x / 2), jnp.cos(x / 2)]])


def RZ(x):
    """
    The parametrized RY-gate

    :param x: the rotation angle
    :type x: float
    """
    return jnp.array([[jnp.exp(-1j * x / 2), 0], [0, jnp.exp(1j * x / 2)]])


def U3_Rot(params):
    """
    General 1Q unitary. Same parametrization method as "U3Gate" from Qiskit.

    :param params: the 3 rotation angles, theta, phi, and lambda.
    :type params: (jax) numpy array / list of float
    """
    t, p, l = params
    gate = jnp.array([[jnp.cos(t / 2), -jnp.exp(1j * l) * jnp.sin(t / 2)],
                      [jnp.exp(1j * p) * jnp.sin(t / 2), jnp.exp(1j * (p + l)) * jnp.cos(t / 2)]])
    return gate


def U3_Pauli(params):
    """
    General 1Q unitary, parametrized by Pauli basis decomposition of the exponent Hermitian matrix.

    :param params: the 3 coefficients of Pauli basis elements (excluding that of Id, which would only contributes to an
                   additional global phase).
    :type params: (jax) numpy array / list of float
    """
    x, y, z = params
    n = jnp.linalg.norm(params)
    a = jnp.cos(n / 2) - 1j * z / n * jnp.sin(n / 2)
    b = -y / n * jnp.sin(n / 2) - 1j * x / n * jnp.sin(n / 2)
    c = y / n * jnp.sin(n / 2) - 1j * x/n * jnp.sin(n / 2)
    return jnp.array([[a, b], [c, jnp.conjugate(a)]])


def GU1(params, parametrization=0):
    """
    A wrapper function for the various functions realizing the general 1Q unitary. There are many ways to parametrize a
    general single qubit unitary with 3 parameters.

    :param params: An array of 3 elements.
    :type params: (jax) numpy array / list of float
    :param parametrization: An integer that defines the parametrization method. Currently, 5 methods are supported.

                            0. Pauli basis parametrization
                            1. Euler angle parametrization, RZ - RY - RZ. Same as "qml.Rot" from Pennylane
                            2. Euler angle parametrization, RY - RZ - RY.
                            3. Euler angle parametrization, RX - RZ - RX.
                            4. Euler angle parametrization, qiskit version
    :type parametrization: int
    """
    if parametrization == 0:  # recommended interval for sampling: [0, 4pi]
        return U3_Pauli(params)
    if parametrization == 1:  # recommended interval for sampling: [0, 4pi]
        return RZ(params[2]) @ RY(params[1]) @ RZ(params[0])
    if parametrization == 2:  # recommended interval for sampling: [0, 4pi]
        return RY(params[2]) @ RZ(params[1]) @ RY(params[0])
    if parametrization == 3:  # recommended interval for sampling: [0, 4pi]
        return RX(params[2]) @ RZ(params[1]) @ RX(params[0])
    if parametrization == 4:  # recommended interval for sampling: [0, 4pi]
        return U3_Rot(params)


#  Fixed 2QG ###########################################################################################################

def CNOT():
    """
    The circuit stack consists of 4 levels, which from bottom to top are: gate, block, layer and circuit.

    Gates are at the bottom level of the stack. Functions in this module realize gates that can be either parametrized
    or unparametrized. Parametrized gates take parameters as input, while unparametrized gates are without input.
    Functions of both kinds output a unitary matrix in jax numpy array form.

    Controlled X gates. Native entangling gate of many superconducting quantum computers.
    """
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])


def CZ():
    """
    Controlled Z gates. Native entangling gate of many superconducting quantum computers.
    """
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]])


#  Parametrized 2QG ####################################################################################################

def GU2(params):
    """
    General 2Q unitary parametrized by 15 coefficients of the 2Q Pauli basis (excluding Id)

    :param params: An array of 15 elements.
    :type params: (jax) numpy array / list of float
    """
    return expm(-0.5j * jnp.tensordot(params, PauliBasis2()[1:], [0, 0]))


#  Others ##############################################################################################################

def Haar_Random(nq, key):
    """
    Randomly generates a general nq-qubit unitary, with respect to Haar measure, by using QR-decomposition.
    Ref: https://arxiv.org/abs/math-ph/0609050


    :param nq: the number of qubits
    :type nq: int
    :param key: the random key
    :type key: jax key
    """
    N = 2 ** nq
    key, subkey = jax.random.split(key)
    M = jax.random.normal(key, shape=(N, N)) + 1j * jax.random.normal(subkey, shape=(N, N))
    Q, R = jnp.linalg.qr(M)
    D = jnp.diagonal(R)
    L = D / jnp.absolute(D)
    U = jnp.multiply(R, L)

    return U
