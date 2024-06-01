from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    'This function read the requirements files and return a list of requi;red lib'

    with open(file_path, 'r') as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name = 'Steam-store-data-ML-project',
    version = '1.0.0',
    author = 'Ashish Singh',
    author_email = 'ashishsingh217070@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('get_requirements.txt')
    )