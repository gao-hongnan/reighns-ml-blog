# setup.py
# Setup installation for the application

from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="reighns-gradual-warmup-scheduler",
    version="0.1",
    license="MIT",
    description="Hongnan G's implementation of gradual warmup scheduler in PyTorch.",
    author="Hongnan G",
    author_email="reighns.sjr.sjh.hrj@gmail.com",
    url="https://github.com/reigHns/reighns-pytorch-gradual-warmup-scheduler.git",
    keywords=[
        "machine-learning",
        "pytorch",
        "scheduler",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=[required_packages],
    dependency_links=[]
)
