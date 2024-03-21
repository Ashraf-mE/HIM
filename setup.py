from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_packages(file_name) -> list:

    with open(file_name) as f:
        packages = f.readlines()
        packages = [package.replace('\n', ' ') for package in packages]

        if HYPHEN_E_DOT in packages:
            packages.remove(HYPHEN_E_DOT)
    return packages

setup(
    name='HIM',
    version='0.0.1',
    author = 'Ashraf',
    author_email='mohammadashrafp10@gmail.com',
    packages= find_packages(),
    install_requires=get_packages('requirements.txt')
)