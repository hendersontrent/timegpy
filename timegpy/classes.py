class Node:
    def __init__(self, op=None, left=None, right=None, lag=None, exponent=None):
        self.op = op          # '+', '-', '*', '/' or None
        self.left = left      # Node or None
        self.right = right    # Node or None
        self.lag = lag        # int or None
        self.exponent = exponent  # float or None