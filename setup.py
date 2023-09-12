from setuptools import setup, find_packages, Extension, Command

setup(
    name="FEM4INAS",
    #Version=__version__,
    description="""FEM4INAS is an aeroelastic toolbox written and parallelized in Python, which acts as a post-processor of commercial software such as MSC Nastran. Arbitrary FE models built for linear aeroelastic analysis are enhanced with geometric nonlinear effects, flight dynamics and linearized state-space solutions about nonlinear equilibrium.""",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    keywords="nonlinear aeroelastic structural aerodynamic analysis",
    author="Alvaro Cea",
    author_email="alvaro_cea",
    url="https://github.com/ACea15/FEM4INAS",
    license="",
    packages=find_packages(
        where='./',
        include=['fem4inas*'],
        exclude=['tests']
        ),
    # data_files=[
    #     ("./lib/UVLM/lib", ["libuvlm.so"]),
    #     ("./lib/xbeam/lib", ["libxbeam.so"])
    #     ],
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        #"PyYAML",
        "ruamel.yaml",
        "jax",
        "jaxlib",
        "diffrax",
        "jaxopt",
        "multipledispatch"
    ],
    extras_require={
        "cpu":[
        "jax[cpu]"],
        "plot": [
            "matplotlib",
            "plotly",
            "pyvista"
            ],
        "docs": [
            "sphinx",
            "myst-parser",
            "sphinx_rtd_theme",
            "nbsphinx"
                 ],
        "all": [
            "pyNastran",
            "jupyterlab",
            "pandas",
            "matplotlib",
            "plotly",
            "pyvista",
            "sphinx",
            "myst-parser",
            "sphinx_rtd_theme",
            "nbsphinx"
                 ],
    })
