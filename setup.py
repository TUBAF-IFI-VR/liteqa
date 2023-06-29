import sys
import glob
from setuptools import setup
import setuptools.command.install
from distutils.core import setup, Extension

def fix_pvpython_interpreter():
	for i in glob.glob("build/scripts*/*.py"):
		text = open(i, "r").read()
		text = text.replace("#!python\n", "#!/usr/bin/env pvpython\n")
		open(i, "w").write(text)

class CustomInstall(setuptools.command.install.install):
	def run(self):
		fix_pvpython_interpreter()
		super().run()

liteqa_c = Extension(
	'liteqa_c',
	define_macros = [],
	include_dirs = [],
	libraries = [],
	library_dirs = [],
	sources = ['liteqa_c/liteqa_c.cpp']
)

setup(
	name='liteqa',
	version='0.1.0',
	description='Lossy In-Situ Tabular Encoding for Query-driven Access',
	url='https://github.com/lehmann7/liteqa',
	author='Henry Lehmann',
	author_email='henry.lehmann@informatik.tu-freiberg.de',
	license='MIT',
	packages=['liteqa'],
	install_requires=[
		'zstd',
		'pyfastpfor',
		'hilbertcurve',
		'numpy',
	],
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research/Visualization',
		'License :: OSI Approved :: BSD License',
		'Operating System :: POSIX :: Linux',
		'Programming Language :: Python :: 3.6',
	],
	ext_modules=[liteqa_c],
	package_data={'liteqa': ['paraview/*']},
	include_package_data=True,
	scripts=["vti2liteqa.py", "lqaquery.py"],
	options={'build_scripts': {'executable': '/usr/bin/pvpython'}},
	cmdclass={'install': CustomInstall},
)
