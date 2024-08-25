from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="findr",
    author="Tim Kim",
    author_email="tdkim@princeton.edu",
    description="Flow-field inference for neural data using deep recurrent networks (FINDR)",
    url="https://github.com/Brody-Lab/findr",
    install_requires=requirements,
    packages=find_packages(),
)