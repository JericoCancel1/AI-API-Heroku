from setuptools import setup

setup(
    name="myapp",
    install_requires=[
        "tensorflow==2.12.0",
        "tensorflow-addons==0.21.0",
        "tensorflow-estimator==2.12.0",
        "tensorflow-io-gcs-filesystem==0.31.0",
    ],
)
