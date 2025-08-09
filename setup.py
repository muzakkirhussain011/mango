# setup.py
from setuptools import setup, find_packages
setup(
    name="faircare-fl",
    version="1.0.1",
    description="FairCare-FL: Next-Gen Fair Federated Learning (Healthcare)",
    packages=find_packages(exclude=("tests", "paper", "runs")),
    include_package_data=True,
)
