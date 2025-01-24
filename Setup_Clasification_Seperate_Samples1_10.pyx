from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("Clasification_Seperate_Samples1_10.pyx")
)
