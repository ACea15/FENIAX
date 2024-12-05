"""Main FENIAX."""

def main(
    input_file: str = None,
    input_dict: dict = None,
    input_obj = None,
    return_driver: bool = False,
    device_count: int = None):
    """Main ``FEM4INAS`` routine


    Parameters
    ----------
    input_file : str
        Path to YAML input file
    input_dict : dict
        Alternatively, dictionary with the settings to be loaded into
        the Config object.
    input_obj : Config
        Alternatively input the Config object directly.

    Returns
    -------
    Solution
        Data object with the numerical solution saved along the
        process.

    """
    if device_count is not None:
        import os
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={device_count}"

    import feniax.drivers
    import feniax.preprocessor.configuration as configuration  # Config, ValidateConfig
    from feniax.preprocessor.solution import Solution

    # from jax.config import config;
    import jax
    jax.config.update("jax_enable_x64", True)
        
    config = configuration.initialise_config(input_file, input_dict, input_obj)
    if config.driver.sol_path is not None:
        configuration.dump_to_yaml(
            config.driver.sol_path / "config.yaml", config, with_comments=True
        )
    Driver = feniax.drivers.factory(config.driver.typeof)
    driver = Driver(config)
    driver.pre_simulation()
    driver.run_case()

    if return_driver:  # return driver object for inspection
        return driver
    else:  # just return the solution data
        return driver.sol.data
