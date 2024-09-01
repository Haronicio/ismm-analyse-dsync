from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options

# Activer les optimisations
Options.inline = True
Options.fastmath = True

setup(
    ext_modules=cythonize(
        # "adsync.pyx",
        "adsync.py",
        compiler_directives={
            'optimize.use_switch': True,  # Utilisation de switch en C
            'optimize.inline_defnode_calls': True  # Inline des fonctions
        }
    ),
    extra_compile_args=["-O3", "-ffast-math",'str3'],
    extra_link_args=["-O3", "-ffast-math"]
)