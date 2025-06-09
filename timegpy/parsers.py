from classes import Node

def tree_to_expression(node: Node) -> str:
    if node.op is None:
        base = f"X_t+{node.lag}"
        if node.exponent is not None:
            base = f"({base})^{node.exponent}"
        return base
    else:
        left_expr = tree_to_expression(node.left)
        right_expr = tree_to_expression(node.right)
        return f"({left_expr} {node.op} {right_expr})"

def tree_to_feature_string(node: Node) -> str:
    return f"mean({tree_to_expression(node)})"