from setuptools import setup, find_packages, Extension, Command
# conda install -c conda-forge libstdcxx-ng
setup(
    name="FEM4INAS",
    #Version=__version__,
    description="""FEM4INAS is an aeroelastic toolbox written and parallelized in Python, which acts as a post-processor of commercial software such as MSC Nastran. Arbitrary FE models built for linear aeroelastic analysis are enhanced with geometric nonlinear effects, flight dynamics and linearized state-space solutions about nonlinear equilibrium.""",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    keywords="nonlinear aeroelastic structural aerodynamic analysis",
    author="Alvaro Cea",
    author_email="alvaro_cea@outlook.com",
    url="https://github.com/ACea15/FEM4INAS",
    license="GPL-3",
    packages=find_packages(
        where='./',
        include=['fem4inas*'],
        exclude=['tests']
        ),
    python_requires=">=3.11",
    # include_package_data=True,
    # package_data={'': ['examples/*']},
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
        "tabulate",
        "multipledispatch",
        "pyNastran"
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
        "tests": [
            "pytest"],
        "streamlit": [
            "streamlit",
            "stpyvista",
            "streamlit-pdf-viewer"],
        "all": [
            "numpy",
            "scipy",
            "pandas",
            "ruamel.yaml",
            "jax",
            "jaxlib",
            "diffrax",
            "jaxopt",
            "tabulate",
            "multipledispatch",
            "pyNastran",            
            ####
            "matplotlib",
            "plotly",
            "pyvista",
            ####
            "pytest",
            ####
            "python-lsp-server",
            "python-lsp-ruff",
            ####
            "jupyterlab",
            "matplotlib",
            "plotly",
            "pyvista",
            ####
            "sphinx",
            "myst-parser",
            "sphinx_rtd_theme",
            "nbsphinx",
            #####
            "streamlit",
            "stpyvista",
            "streamlit-pdf-viewer"
                 ],
    })
