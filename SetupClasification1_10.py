from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("Clasification1_10.pyx")
)
