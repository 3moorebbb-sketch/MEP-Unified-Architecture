from setuptools import setup, find_packages

setup(
    name='mep-suite',
    version='3.1.0', # V3.1 Update: Refactored optimizer to Riemannian Momentum based on ablation data, fixed hardware trig quadrature, and updated Master Equation tensor projections.
    packages=find_packages(),
    description='Mandelbrot-Euler-Planck (MEP) Architecture Software Suite. Physics-based, continuous-wave solvers and optimizers.',
    long_description=open('README.md').read() if open('README.md').read() else '',
    long_description_content_type='text/markdown',
    author='3MOORE. BBB',
    url='https://github.com/yourusername/mep-architecture', # Remember to update this URL!
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: Public Domain',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    keywords='machine learning, thermodynamics, physics, optimization, kuramoto, langevin',
)
