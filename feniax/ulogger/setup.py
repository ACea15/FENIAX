import logging
import pathlib

def set_logging(config):


    # integrate with warning module
    logging.captureWarnings(True)
    
    # Create a root logger
    root_logger = logging.getLogger()
    # Prevent adding multiple handlers
    if root_logger.hasHandlers():
        reset_logging()
        
    root_logger.setLevel(getattr(logging, config.log.level.upper()))

    # Create a console handler with a console-specific format
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(name)s | %(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    if config.log.path is not None:
        log_file = config.log.path / f'{config.log.file_name}.log'
    else:
        log_file = config.driver.sol_path / f'{config.log.file_name}.log'
    file_handler = logging.FileHandler(log_file, mode=config.log.file_mode)
    file_formatter = logging.Formatter("%(name)s (L%(lineno)s): %(asctime)s | %(levelname)s | P %(process)d | %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add the handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

def reset_logging():
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove all handlers associated with the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Reset logging configuration
    logging.basicConfig(level=logging.NOTSET, handlers=[])
        
def get_logger(name):

    logger = logging.getLogger(name)
    return logger
