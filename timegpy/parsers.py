import re
from .classes import Node
    
#---------------- String to printed tree -----------------

def strip_mean(expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("mean(") and expr.endswith(")"):
        return expr[len("mean("):-1].strip()
    return expr

def tokenize(expr: str):
    TOKEN_REGEX = r'sin|cos|tan|X_t(?:[\+\-]\d+)?|\d+(?:\.\d+)?|\^|[\+\-\*/()]'
    return re.findall(TOKEN_REGEX, expr.replace(' ', ''))

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def parse(self):
        return self.parse_expression()

    def parse_expression(self, min_precedence=1):
        node = self.parse_term()
        while True:
            op = self.peek()
            prec = self.get_precedence(op)
            if prec < min_precedence:
                break
            self.consume()
            right = self.parse_expression(prec + (0 if op == '^' else 1))
            node = Node(op=op, left=node, right=right)
        return node

    def parse_term(self):
        tok = self.peek()
        if tok in ('sin', 'cos', 'tan'):
            self.consume()
            if self.consume() != '(':
                raise ValueError("Expected '(' after unary operator")
            child = self.parse_expression()
            if self.consume() != ')':
                raise ValueError("Expected ')' after unary operator argument")
            return Node(op=tok, left=child)
        
        if tok == '(':
            self.consume()
            node = self.parse_expression()
            if self.consume() != ')':
                raise ValueError("Expected ')'")
            return node
        elif re.match(r'X_t(?:[\+\-]\d+)?', tok):
            self.consume()
            lag_match = re.search(r'[\+\-]\d+', tok)
            lag = int(lag_match.group()) if lag_match else 0
            return Node(lag=lag)
        elif re.match(r'\d+(?:\.\d+)?', tok):
            self.consume()
            return Node(value=float(tok))
        else:
            raise ValueError(f"Unexpected token: {tok}")

    def get_precedence(self, op):
        if op == '^': return 3
        if op in ('*', '/'): return 2
        if op in ('+', '-'): return 1
        return -1

def parse_expression(expr: str) -> Node:
    cleaned = strip_mean(expr)
    tokens = tokenize(cleaned)
    parser = Parser(tokens)
    return parser.parse()

def represent(expr: str):
    def _print_tree(node, indent="", is_left=True):
        if node is None:
            return

        prefix = indent + ("├── " if is_left else "└── ")

        if node.is_operator():
            if node.op in ('sin', 'cos', 'tan'):
                print(prefix + node.op)
                _print_tree(node.left, indent + ("│   " if is_left else "    "), True)
            else:
                print(prefix + node.op)
                _print_tree(node.left, indent + ("│   " if is_left else "    "), True)
                _print_tree(node.right, indent + ("│   " if is_left else "    "), False)

        elif node.is_lag_term():
            label = f"X_t{node.lag:+}" if node.lag != 0 else "X_t"
            if node.exponent is not None:
                print(prefix + "^")
                _print_tree(Node(lag=node.lag), indent + ("│   " if is_left else "    "), True)
                _print_tree(Node(value=node.exponent), indent + ("│   " if is_left else "    "), False)
            else:
                print(prefix + label)

        elif node.is_constant():
            print(prefix + str(node.value))

    tree = parse_expression(expr)
    _print_tree(tree, indent="", is_left=False)

#---------------- Tree to string -----------------

def tree_to_expression(node):
    if node.is_binary_operator():
        left_str = tree_to_expression(node.left)
        right_str = tree_to_expression(node.right)
        return f"({left_str} {node.op} {right_str})"
    elif node.is_unary_operator():
        child_str = tree_to_expression(node.left)
        return f"{node.op}({child_str})"
    elif node.is_constant():
        return f"{node.value:.4f}"
    elif node.is_lag_term():
        base = f"X_t+{node.lag}"
        if node.exponent is not None:
            return f"({base} ^ {node.exponent})"
        else:
            return base
    else:
        raise ValueError("Invalid node encountered in tree.")
    
def tree_to_feature_string(node: Node) -> str:
    return f"mean({tree_to_expression(node)})"
