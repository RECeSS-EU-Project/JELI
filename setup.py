from setuptools import setup, find_packages

NAME = "jeli"
VERSION = "9999"

setup(name=NAME,
    version=VERSION,
    author="Clémence Réda",
    author_email="recess-project@proton.me",
    url="https://github.com/RECeSS-EU-Project/jeli",
    license_files = ('LICENSE'),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    keywords='',
    description="Package for Joint Embedding-classifier Learning for Interpretability. Learns feature/item/user embeddings with specific structures, recommends new item-user associations and provides feature importance scores.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={'':"src"},
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "pykeen==1.10.1",
        "stanscofi>=2.0.1",
        "scikit-learn>=1.2.2",
        "scipy>=1.9.3",
        "torch_geometric>=2.4.0",
        "igraph>=0.11.4",
        "torch>=2.1.2",
        "tqdm>=4.66.2",
        "qnorm>=0.8.1",
        "statsmodels>=0.14.1",
        "typing_extensions>=4.2.0",
    ],
    entry_points={},
)
