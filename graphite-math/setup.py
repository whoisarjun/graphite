from setuptools import setup, find_packages

setup(
    name='graphite-math',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    package_data={
        'graphite': ['empty.json']
    }
)