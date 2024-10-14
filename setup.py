import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="randomcov",
    version="0.0.3",
    description="Random covariance generation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/randomcov",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=["randomcov",
              "randomcov.corrgens",
              "randomcov.vargens"
              ],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=['numpy','scikit-learn','scipy'],
    entry_points={
        "console_scripts": [
            "randomcov=randomcov.__main__:main",
        ]
    },
)
