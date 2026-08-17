from setuptools import setup, find_packages
import os

# Read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mep-architecture-suite', # <-- CHANGED TO A UNIQUE NAME
    version='5.0.0',
    packages=find_packages(),
    description='Mandelbrot-Euler-Planck (MEP) Architecture: Physics-based PyTorch optimizers and topological memory substrates.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='3MOORE. BBB',
    url='https://github.com/yourusername/mep-architecture',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'torchvision>=0.15.0'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    keywords='machine learning, thermodynamics, physics, optimization, continual learning, jacobian',
)
