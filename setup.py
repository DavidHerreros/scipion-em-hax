"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from pathlib import Path
from flexutils import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
long_description = Path(path.join(here, 'README.md')).read_text()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='scipion-em-hax',  # Required
    version=__version__,  # Required
    description='Plugin to execute Hax package inside Scipion',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',
    url='https://github.com/scipion-em/scipion-em-hax',  # Optional
    author='David Herreros',  # Optional
    author_email='dherreros@cnb.csic.es',  # Optional
    keywords='scipion continuous-heterogeneity imageprocessing scipion-3.0 Hax',  # Optional
    packages=find_packages(),
    install_requires=[requirements],
    entry_points={'pyworkflow.plugin': 'hax = hax'},
    package_data={  # Optional
       'hax': ['logo.png'],
    }
)
