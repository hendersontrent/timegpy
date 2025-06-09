from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()


install_requires = [
        'numpy',
        'random',
        'operator',
        'pandas',
        'matplotlib',
        're',
        'copy',
        'typing',
        'scikit-learn',
        'scipy',
        'tqdm'
]

setup(
    name='timegpy',
    packages=find_packages(),
    version='0.1.0',
    description='Find informative time-average features using genetic programming',
    author='Trent Henderson',
    author_email='then6675@uni.sydney.edu.au',
    url='https://github.com/hendersontrent/timegpy',
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ]
    install_requires=install_requires
)