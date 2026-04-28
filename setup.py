from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="mmlandmarks",
    version="1.0.0",
    description="MMLandmarks: a Cross-View Instance-Level Benchmark for Geo-Spatial Understanding",
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Oshkr/mml-codebase",
    author="Oskar Kristoffersen",
    author_email="ofhkr@dtu.dk",
    license="MIT",
    install_requires=requirements,
    python_requires=">=3.9",
    include_package_data=True,
)
