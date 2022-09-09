from setuptools import find_packages, setup

install_requires = [
    "accelerate==0.12.0",
    "diffusers==0.3.0",
    "ftfy==6.1.1"
    "gradio==3.2",
    "torch==1.12.0",
    "torchvision==0.13.0",
    "tqdm",
    "transformers==4.21.1"
]

setup(
    name="textual-inversion",
    version="0.0",
    packages=find_packages(),
    install_requires=install_requires
)
