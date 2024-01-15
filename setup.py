import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ppyPatternRecognition",
    version="0.0.1",
    author="Jirayuwat Boonchan",
    author_email="jirayuwat.dev@gmail.com",
    description="Pattern Recognition Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[line.rstrip() for line in open('requirements.txt')],
)