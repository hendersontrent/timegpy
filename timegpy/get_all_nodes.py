from classes import Node

def get_all_nodes(node: Node, include_root=False) -> List[Node]:
    nodes = []
    def recurse(n):
        if n is None:
            return
        if n.op is not None:
            recurse(n.left)
            recurse(n.right)
        nodes.append(n)
    recurse(node)
    return [node] + nodes if include_root else nodes

def replace_node(tree: Node, target: Node, replacement: Node):
    def recurse(n):
        if n.left is target:
            n.left = replacement
        elif n.right is target:
            n.right = replacement
        else:
            if n.left: recurse(n.left)
            if n.right: recurse(n.right)
    recurse(tree)