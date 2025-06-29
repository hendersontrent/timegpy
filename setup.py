from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'pingouin',
    'scipy',
    'tqdm'
]

docs_extras = [
    'sphinx >= 3.0.0',  # Force RTD to use >= 3.0.0
    'sphinx-rtd-theme >= 1.1.1'
]

setup(
    name='timegpy',
    license="MIT",
    packages=find_packages(),
    version='0.1.0',
    description='Find informative time-average features using genetic programming',
    author='Trent Henderson',
    author_email='then6675@uni.sydney.edu.au',
    url='https://github.com/hendersontrent/timegpy',
    long_description=long_description,
    keywords=[
        "time series",
        "classification",
        "time series features",
        "features",
        "genetic algorithm",
        "genetic programming"
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix"
        "Operating System :: Microsoft :: Windows"
        "Operating System :: MacOS"
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    install_requires=install_requires,
    extras_require={'docs': docs_extras}
)