"""FEM4INAS main"""
import argparse
import fem4inas.drivers
from fem4inas.preprocessor.inputs import Config

def main(input_file=None, input_dict=None, input_obj=None):
    """
    Main ``FEM4INAS`` routine

    """
    if input_dict is None and input_obj is None:
        parser = argparse.ArgumentParser(prog='FEM4INAS', description=
        """This is the executable of Fininte-Element Models for Intrinsic Nonlinear Aeroelastic Simulations.""")
        parser.add_argument('input_file', help='path to the *.yml input file',
                            type=str, default='')
        if input_file is not None:
            args = parser.parse_args(input_file)
        else:
            args = parser.parse_args()
            
    elif input_dict is not None and (input_file is None and
                                     input_obj is None):
        config = Config()

    elif input_dict is not None and (input_file is None and
                                     input_obj is None):
        config = input_obj

    Driver = fem4inas.drivers.factory(config.engine)
    driver = Driver(config)
    driver.pre_simulation()
    driver.run_cases()

