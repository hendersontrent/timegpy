def ensure_has_X_t(node):
    """
    Recursively check if a tree contains at least one terminal with lag == 0
    """
    if node is None:
        return False
    if node.op is None:
        return node.lag == 0
    return ensure_has_X_t(node.left) or ensure_has_X_t(node.right)