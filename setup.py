from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read()

setup(
    name="maze_runner",
    version="0.1",
    packages=find_packages(),
    install_requires=REQUIREMENTS
)
