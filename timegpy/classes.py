class Node:
    def __init__(self, op=None, left=None, right=None, lag=None, exponent=None, value=None):
        self.op = op                # '+', '-', '*', '/', 'sin', 'cos', 'tan', or None
        self.left = left            # Node or None
        self.right = right          # Node or None
        self.lag = lag              # int or None
        self.exponent = exponent    # float or None
        self.value = value          # float or None

    def is_constant(self):
        return self.op is None and self.lag is None and self.value is not None

    def is_lag_term(self):
        return self.op is None and self.lag is not None

    def is_binary_operator(self):
        return self.op in ('+', '-', '*', '/')

    def is_unary_operator(self):
        return self.op in ('sin', 'cos', 'tan')
