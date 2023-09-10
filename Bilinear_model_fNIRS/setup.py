from setuptools import setup, find_packages
import os

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="Bilinear_model_fNIRS",
    version='0.1.0',
    description='fNIRS Synthetic data generator based on the Bilinear Model [S.Tak 2015] ',
    long_description_content_type="text/markdown",
    author="Mario De Los Santos",
    author_email="madlsh3517@gmail.com",
    url='https://github.com/yourusername/your_project_name',
    install_requires=requirements,
    packages=find_packages(exclude=('tests*', 'docs')),
    license='MIT', 
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8', 
)

