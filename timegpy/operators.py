import operator

OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': lambda a, b: a / b if b != 0 else 1,
    '^': lambda a, b: a ** b,
}
OPERATOR_ARITY = {op: 2 for op in OPERATORS}