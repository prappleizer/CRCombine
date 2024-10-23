from setuptools import find_packages, setup

setup(
    name="crcombine",
    version="0.1.1",
    author="Imad Pasha",
    author_email="imad.pasha@yale.edu",
    description="A Python package for cr replacement",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/prappleizer/rejectCR",
    download_url="https://github.com/prappleizer/rejectCR/archive/refs/tags/v0.1.0-beta.tar.gz",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "astropy",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "crcombine = crcombine.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
