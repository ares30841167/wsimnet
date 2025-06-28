def detect_type_from_string(input_str: str) -> str:
    # Try to convert to a boolean
    if input_str.lower() == 'true':
        return 'bool'
    elif input_str.lower() == 'false':
        return 'bool'

    # Try to convert to an integer
    try:
        int(input_str)
        if ('.' in input_str):
            raise ValueError('String contains decimal point')

        return 'int'
    except ValueError:
        pass

    # Try to convert to a float
    try:
        float(input_str)

        return 'float'
    except ValueError:
        pass

    # Try to detect if it's a list
    if input_str.startswith('[') and input_str.endswith(']'):
        return 'list'

    # Try to detect if it's a dictionary
    if input_str.startswith('{') and input_str.endswith('}'):
        return 'dict'

    # Try to detect if it's a tuple
    if input_str.startswith('(') and input_str.endswith(')'):
        return 'tuple'

    # If none of the above, return it as a string
    return 'str'
