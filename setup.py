import setuptools
import os
import sys

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version must be >=3.7.")

with open("./requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extras = {}
extras["docs"] = [
    "recommonmark",
    "nbsphinx",
    "sphinx-autobuild",
    "sphinx-rtd-theme",
    "sphinx-markdown-tables",
    "sphinx-copybutton",
]

extras["augly"] = ["augly==0.1.10"]
extras["textattack"] = ["textattack[tensorflow]"]
extras["art"] = ["adversarial-robustness-toolbox==1.8"]
extras["dev"] = extras["docs"] + extras["augly"] + extras["textattack"] + extras["art"]

setuptools.setup(
    name="counterfit",
    maintainer="Counterfit Developers",
    version="1.1.0",
    author="Azure Trustworthy Machine Learning",
    description="Counterfit project to simulate attacks on ML systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Azure/counterfit",
    classifiers=[
        "Development Status :: development",
        "Intended Audience :: security/ML/research",
        "Programming Language :: Python :: 3.7+",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_namespace_packages(
        exclude=["build", "docs", "dist", "tests", "infrastructure", "examples"]
    ),
    install_requires=required,
    python_requires=">=3.7",
    extras_require=extras,
    entry_points={
        "console_scripts": ["counterfit=counterfit.terminal:main"],
    },
)
