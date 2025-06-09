from classes import Node

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