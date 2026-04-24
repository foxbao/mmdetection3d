def print_dict_keys(d, prefix=''):
    if not isinstance(d, dict):
        return

    for k, v in d.items():
        full_key = f'{prefix}.{k}' if prefix else str(k)
        print(full_key)
        if isinstance(v, dict):
            print_dict_keys(v, full_key)