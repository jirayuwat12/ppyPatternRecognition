import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_requirements(fname : str):
    '''
    extract library name from `fname`
    '''
    with open(fname) as fp:
        reqs = list()
        for lib in fp.read().split("\n"):
            # Ignore pypi flags and comments
            if not lib.startswith("-") or lib.startswith("#"):
                reqs.append(lib.strip())
        return reqs


install_requires = get_requirements("requirements.txt")

setuptools.setup(
    name="ppyPatternRecognition",
    version="0.0.0.3",
    author="Jirayuwat Boonchan",
    author_email="jirayuwat.dev@gmail.com",
    description="Pattern Recognition Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=install_requires
)