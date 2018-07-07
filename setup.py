import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bignmf",
    version="1.0.2",
    author="Haran Rajkumar, Vaibhav Kulshrestha",
    author_email="haranrajkumar97@gmail.com, vaibhav1kulshrestha@gmail.com",
    description="Non-negative matrix factorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thenmf/bignmf",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'numpy',
        'pandas',
        'fastcluster',
        'scipy'
    ]

)