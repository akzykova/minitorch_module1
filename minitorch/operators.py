"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Return the product of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The result of x * y.
    """
    return x * y


def id(x: float) -> float:
    """Return the input value unchanged.

    Args:
        x (float): Input number.

    Returns:
        float: The same value as x.
    """
    return x


def add(x: float, y: float) -> float:
    """Return the sum of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The result of x + y.
    """
    return x + y


def neg(x: float) -> float:
    """Return the negation of a number.

    Args:
        x (float): Input number.

    Returns:
        float: The value -x.
    """
    return -1.0 * x

def lt(x: float, y: float) -> float:
    """Check if x is less than y.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: 1.0 if x < y, otherwise 0.0.
    """
    return x < y


def eq(x: float, y: float) -> float:
    """Check if two numbers are equal.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: 1.0 if x == y, otherwise 0.0.
    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the greater of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The maximum of (x, y).
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close within a tolerance of 1e-2.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: 1.0 if |x - y| < 1e-2, otherwise 0.0.
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid activation function.

    The sigmoid function is defined as:
        f(x) = 1 / (1 + exp(-x))

    For numerical stability, it is computed as:
        f(x) = 1 / (1 + exp(-x)) if x >= 0
        f(x) = exp(x) / (1 + exp(x)) if x < 0

    Args:
        x (float): Input value.

    Returns:
        float: The sigmoid of x.
    """
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU activation function.

    The ReLU function is defined as:
        f(x) = x if x > 0 else 0

    Args:
        x (float): Input value.

    Returns:
        float: The ReLU of x.
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm with numerical stability.

    Args:
        x (float): Input value.

    Returns:
        float: log(x + EPS), where EPS is a small constant to avoid log(0).
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
        x (float): Input value.

    Returns:
        float: exp(x).
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the backward pass for the log function.

    If f(x) = log(x), then f'(x) = 1 / x.
    This function returns d * f'(x).

    Args:
        x (float): Input value.
        d (float): Upstream gradient.

    Returns:
        float: Gradient with respect to x.
    """
    return (1 / x) * d


def inv(x: float) -> float:
    """Compute the reciprocal of a number.

    Args:
        x (float): Input value.

    Returns:
        float: 1 / x.
    """
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """Compute the backward pass for the inverse function.

    If f(x) = 1 / x, then f'(x) = -1 / x².
    This function returns d * f'(x).

    Args:
        x (float): Input value.
        d (float): Upstream gradient.

    Returns:
        float: Gradient with respect to x.
    """
    return - (1 / x ** 2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the backward pass for the ReLU function.

    If f(x) = ReLU(x), then f'(x) = 1 if x > 0, else 0.
    This function returns d * f'(x).

    Args:
        x (float): Input value.
        d (float): Upstream gradient.

    Returns:
        float: Gradient with respect to x.
    """
    return (x > 0) * d


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    return lambda lst: [fn(x) for x in lst]


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate each element in a list.

    Uses `map` together with the `neg` function.

    Args:
        ls (Iterable[float]): A list (or iterable) of numbers.

    Returns:
        Iterable[float]: A new iterable where each element is the negation of the corresponding element in ls.
    """
    mapper = map(neg)
    return mapper(ls)

def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    return lambda lst1, lst2: [fn(x, y) for x, y in zip(lst1, lst2)]


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements of two lists.

    Uses `zipWith` together with the `add` function.

    Args:
        ls1 (Iterable[float]): First list of numbers.
        ls2 (Iterable[float]): Second list of numbers.

    Returns:
        Iterable[float]: A new iterable containing the elementwise sums of ls1 and ls2.
    """
    zipper = zipWith(add)
    return zipper(ls1, ls2)

def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    def reducer(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    
    return reducer


def sum(ls: Iterable[float]) -> float:
    """Compute the sum of a list.

    Uses `reduce` with the `add` function.

    Args:
        ls (Iterable[float]): A list (or iterable) of numbers.

    Returns:
        float: The sum of all elements in ls.
    """
    reducer = reduce(fn = add, start = 0)
    return reducer(ls)


def prod(ls: Iterable[float]) -> float:
    """Compute the product of a list.

    Uses `reduce` with the `mul` function.

    Args:
        ls (Iterable[float]): A list (or iterable) of numbers.

    Returns:
        float: The product of all elements in ls.
    """
    reducer = reduce(fn = mul, start = 1.0)
    return reducer(ls)
