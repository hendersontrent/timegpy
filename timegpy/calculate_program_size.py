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