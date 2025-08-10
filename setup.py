from setuptools import setup, find_packages

setup(
    name="faircare",
    version="1.0.0",
    description="FairCare-FL: Unified Fair Federated Learning for Healthcare",
    author="Your Name",
    packages=find_packages(exclude=("tests", "paper")),
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.9",
)
