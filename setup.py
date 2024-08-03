from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requiremnets(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
       requirements=file_obj.readlines()
       requirements=[req.replace("\n","") for req in requirements]

       if HYPEN_E_DOT in requirements:
           requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='my_ml_project',
    version='0.1',
    packages=find_packages(),
    install_requires=get_requiremnets('requirements.txt'),
    description='ML project',
    author='Hatim',
    author_email='aneeshatim3@gmail.com'
)
