from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="findr",
    author="Timothy Doyeon Kim",
    author_email="timothy.doyeon.kim@gmail.com",
    description="Flow-field inference for neural data using deep recurrent networks (FINDR)",
    url="https://github.com/Brody-Lab/findr",
    install_requires=requirements,
    packages=find_packages(),
)