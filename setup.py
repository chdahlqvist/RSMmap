#!/usr/bin/env python3

#!/usr/bin/env python

import os
from setuptools import setup
try:  # pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # pip <= 9.0.3
    from pip.req import parse_requirements
from setuptools.command.install import install


# Hackishly override of the install method
class InstallReqs(install):
    def run(self):
        print(" ********************** ")
        print(" ** Installing PyRSM ** ")
        print(" ********************** ")
        os.system('pip install -r requirements.txt')
        install.run(self)


def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)


# parse_requirements() returns generator of pip.req.InstallRequirement objects
reqs = parse_requirements(resource('requirements.txt'), session=False)
try:
    reqs = [str(ir.req) for ir in reqs]
except:
    reqs = [str(ir.requirement) for ir in reqs]


with open(resource('README.md')) as readme_file:
    README = readme_file.read()


setup(
    name='PyRSM',
    version="1.1.2",
    description='Package for exoplanet detection and characterization via the RSM map algorithm (optimal parametrization via auto-RSM framework)',
    long_description=README,
    license='MIT',
    author='Carl-Henrik Dahlqvist',
    author_email='ch.dahlqvist@gmail.com',
    url='https://github.com/chdahlqvist/RSMmap',
    cmdclass={'install': InstallReqs},
    packages=['PyRSM'],
    install_requires=reqs,
    zip_safe=False,
    classifiers=['Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: POSIX :: Linux',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Scientific/Engineering :: Astronomy'
                 ]
)

