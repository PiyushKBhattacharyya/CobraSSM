from setuptools import setup, find_packages

setup(
    name="cobrassm",
    version="0.1.0",
    description="CobraSSM: A novel sequence model combining Multi-Scale SSMs and Differentiable Sparse Memory",
    author="Antigravity",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)
