#from distutils.core import setup, Extension
from setuptools import find_packages, setup, Extension

module1 = Extension('cppvoro',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '1')],
                    include_dirs = [
                        '/usr/include/voro++',
                        '/usr/lib/python3.7/site-packages/numpy/core/include'],
                    libraries = ['voro++', 'CGAL_Core', 'CGAL', 'gmp'],
                    library_dirs = ['/usr/lib'],
                    sources = ['src/vorosym/cppvoro.cpp', 'src/vorosym/rotationsym.cpp'])

setup (name = 'vorosym',
       version = '0.0.2',
       description = 'Python library for Voronoi diagrams with periodic boundary conditions and symmetry measures',
       author = 'Peter Bjørn Jørgensen',
       author_email = 'peterbjorgensen@gmail.com',
       #url = 'https://docs.python.org/extending/building',
       package_dir={'': 'src'},
       packages=find_packages('src'),
       ext_modules=[module1])
