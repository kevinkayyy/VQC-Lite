import jax.numpy as jnp


def fidelity(psi1, psi2):
    """
    Compute the fidelity between 2 statevectors.

    :param psi1: the first statevector
    :type psi1: (jax) numpy array
    :param psi2: the second statevector
    :type psi2: (jax) numpy array
    :return: fidelity
    :rtype: float
    """
    return jnp.abs(jnp.sum(psi1 * psi2.conjugate())) ** 2


def distance(psi1, psi2):
    """
    Compute the coordinate-wise distance between 2 statevectors.

    :param psi1: the first statevector
    :type psi1: (jax) numpy array
    :param psi2: the second statevector
    :type psi2: (jax) numpy array
    :return: coordinate-wise distance
    :rtype: float
    """
    return jnp.abs(jnp.sqrt(jnp.sum(jnp.abs(psi1 - psi2) ** 2)))
