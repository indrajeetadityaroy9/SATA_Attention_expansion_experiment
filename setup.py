from setuptools import setup, find_packages

setup(
    name='sata-attention',
    version='0.1.0',
    description='Self-attention at constant cost per token via symmetry-aware Taylor approximation',
    packages=find_packages(),
    python_requires='>=3.12',
    install_requires=[
        'torch>=2.0',
    ],
)
