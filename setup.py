from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="pyhermes",
    version="0.0.7",  
    description="Hermes - A python package towards an ultimate high-performance algorithm for cosmic statistics of large data sets",
    long_description=long_description,  
    long_description_content_type="text/markdown",
    author="PyHermes Team", 
    author_email="dingdluan@gmail.com", 
    url="https://github.com/PyHermes/PyHermes", 
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    license="MIT",
    keywords="cosmology nbody large-structure statistics correlation-function", 
)

