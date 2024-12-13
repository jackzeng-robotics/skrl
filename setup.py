from setuptools import setup, find_packages

setup(
    name="skrl",
    version="1.3.0",
    description="Modular and flexible library for reinforcement learning on PyTorch and JAX",
    author="Toni-SM",
    author_email="",
    license="MIT License",
    python_requires=">=3.6",
    keywords=[
        "reinforcement-learning",
        "machine-learning",
        "reinforcement",
        "machine",
        "learning",
        "rl",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        "gym",
        "gymnasium",
        "tqdm",
        "packaging",
        "tensorboard",
    ],
    extras_require={
        "torch": ["torch>=1.9"],
        "jax": ["jax>=0.4.3", "jaxlib>=0.4.3", "flax", "optax"],
        "all": ["torch>=1.9", "jax>=0.4.3", "jaxlib>=0.4.3", "flax", "optax"],
        "tests": ["pytest", "hypothesis"],
    },
)
