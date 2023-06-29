import sys
import glob
from setuptools import setup
import setuptools.command.install
from distutils.core import setup, Extension

class CustomInstall(setuptools.command.install.install):
	def run(self): super().run()

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
	url='https://github.com/TUBAF-IFI-VR/liteqa',
	author='Henry Lehmann',
	author_email='lehmann.henry@gmail.com',
	license='APACHE-2.0',
	packages=['liteqa'],
	install_requires=[
		'zstd',
		'pyfastpfor',
		'hilbertcurve',
		'numpy'
	],
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research/Visualization',
		'License :: OSI Approved :: BSD License',
		'Operating System :: POSIX :: Linux',
		'Programming Language :: Python :: 3.10',
	],
	ext_modules=[liteqa_c],
	scripts=["vti2liteqa.py", "lqaquery.py"],
	cmdclass={'install': CustomInstall},
)
