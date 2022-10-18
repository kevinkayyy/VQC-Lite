import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='vqc_lite',
    version='0.1.0',
    author='Kevin Shen',
    author_email='kevinshen.abcd@gmail.com',
    description='vqc_lite: A reader-friendly Python implementation of Variational Quantum Circuits (VQC) based on Jax '
                'and Pennylane',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kevinkayyy/VQC-Lite',
    license='Apache-2.0',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib', 'pennylane', 'jax', 'jaxlib',  'optax', 'tqdm']
)
