def calculate_program_size(node):
    if node.op is None:
        exponent_count = 1 if node.exponent is not None else 0
        return 1 + exponent_count  # one lag term + optional exponent
    else:
        return (
            1 +  # the operator
            calculate_program_size(node.left) +
            calculate_program_size(node.right)
        )
    
#def calculate_program_size(node):
#    if node.op is None:
#        # Terminal: lag or constant, both count as 1
#        is_terminal = node.lag is not None or node.exponent is not None or isinstance(node.op, str)
#        assert is_terminal  # ensures the structure is valid
#        exponent_count = 1 if node.exponent is not None else 0
#        return 1 + exponent_count
#    else:
#        return 1 + calculate_program_size(node.left) + calculate_program_size(node.right)
