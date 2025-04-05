from setuptools import setup, find_packages

setup(
    name='pytorch_replica',
    version='7.0.0', 
    description='An implementation of a PyTorch-like framework from scratch for ANN.',
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',
    author='AxRy', 
    author_email='prajwalsingh7512@gmail.com',  
    url='https://github.com/yourusername/pytorch_replica', 
    packages=find_packages(), 
    install_requires=[
        'numpy',
        'tensorflow',  
        'pandas',  
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'pytorch_replica=pytorch_replica.main:main',  # Command line tool entry
        ],
    },
    python_requires='>=3.6',  # Minimum Python version
)
