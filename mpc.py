def share(x):
    x0 = random.randrange(Q)
    x1 = random.randrange(Q)
    x2 = (x - x0 - x1) % Q
    return [x0, x1, x2]

def reconstruct(shares):
    return sum(shares) % Q


def add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]

def sub(x, y):
    return [ (xi - yi) % Q for xi, yi in zip(x, y) ]


def generate_mul_triple():
    a = random.randrange(Q)
    b = random.randrange(Q)
    c = (a * b) % Q
    return a, b, c

def mul(x, y):
    a, b, c = generate_mul_triple()
    # local masking followed by communication of the reconstructed values
    d = reconstruct(x - a)
    e = reconstruct(y - b)
    # local re-combination
    return d.mul(e) + d.mul(b) + a.mul(e) + c


d * e == xy - xb - ay + ab,
a * e == ay - ab,
b * d == bx - ab
