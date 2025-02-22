from setuptools import setup, find_packages

setup(
    name="ml_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "pytest>=6.2.5",
        "joblib>=1.0.1",
    ],
) 