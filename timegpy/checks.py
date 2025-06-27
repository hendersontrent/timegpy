def ensure_has_X_t(node):
    if node is None:
        return False
    if node.op is None:
        return node.lag == 0
    return ensure_has_X_t(node.left) or ensure_has_X_t(node.right)