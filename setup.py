import setuptools


setuptools.setup(
    name="padm_final_project",
    version="0.0.1",
    description="Package containing support code for BayesNet tutorial",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=["pandas", "networkx", "matplotlib"],
)
