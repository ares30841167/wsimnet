import json


def read_config(config_path: str) -> dict[any, any]:
    # Load the config file
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    return config
