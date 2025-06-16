class Node:
    def __init__(self, op=None, left=None, right=None, lag=None, exponent=None, value=None):
        """
        Represents a node in a symbolic expression tree.

        Parameters:
        - op: Operator as string ('+', '-', '*', '/', or None)
        - left: Left child Node
        - right: Right child Node
        - lag: Integer time lag (if applicable)
        - exponent: Optional exponent applied to lag term
        - value: Float constant (only used if lag is None and op is None)
        """
        self.op = op                # '+', '-', '*', '/', or None
        self.left = left            # Node or None
        self.right = right          # Node or None
        self.lag = lag              # int or None
        self.exponent = exponent    # float or None
        self.value = value          # float or None

    def is_constant(self):
        return self.op is None and self.lag is None and self.value is not None

    def is_lag_term(self):
        return self.op is None and self.lag is not None

    def is_operator(self):
        return self.op is not None
