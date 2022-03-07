from setuptools import setup, find_packages

setup(
    name="mec",
    version="1.0", 
    packages=find_packages(),
    install_requires=["numpy>=1.18.5", 
                      "scikit-learn>=0.23.2", 
                      "scipy>=1.5.2", 
                      "tensorflow_gpu==2.0.1",
                      "tqdm>=4.49.0",
                      "xarray>=0.15.1"],
    python_requires=">=3.6"
)
