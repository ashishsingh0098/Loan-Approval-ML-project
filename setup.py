from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """This function reads the requirements file and returns a list of required libraries."""

    with open(file_path, 'r') as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='Loan-Approval-ML-project',
    version='1.0.0',
    author='Ashish Singh',
    author_email='ashishsingh217070@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
