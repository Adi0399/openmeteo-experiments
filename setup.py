from setuptools import setup, find_packages

setup(
    name="oml-skies",
    version="0.1.0",
    description="Open Meteo ML utils for Sydney rain and precipitation models",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "requests",
        "scikit-learn",
        "joblib",
        "numpy",
        "python-dateutil",
    ],
)
