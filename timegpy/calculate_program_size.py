def calculate_program_size(node):
    if node.is_binary_operator():
        return 1 + calculate_program_size(node.left) + calculate_program_size(node.right)
    
    elif node.is_unary_operator():
        return 1 + calculate_program_size(node.left)
    
    elif node.is_lag_term() or node.is_constant():
        exponent_count = 1 if node.exponent is not None else 0
        return 1 + exponent_count  # 1 for the lag or constant, plus optional exponent
    
    else:
        return 0  # Should not happen if tree is well-formed

