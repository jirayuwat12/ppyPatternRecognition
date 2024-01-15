import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ppyPatternRecognition",
    version="0.0.1.4",
    author="Jirayuwat Boonchan",
    author_email="jirayuwat.dev@gmail.com",
    description="Pattern Recognition Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jirayuwat12/ppyPatternRecognition",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    
)