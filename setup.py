from setuptools import setup, find_packages

setup(
    name="elysia",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "click",
        # add other dependencies here
    ],
) 