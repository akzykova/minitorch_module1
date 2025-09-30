from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import defaultdict

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    delta_vals_plus = list(vals).copy()
    delta_vals_plus[arg] += epsilon

    delta_vals_minus = list(vals).copy()
    delta_vals_minus[arg] -= epsilon

    return (f(*delta_vals_plus) - f(*delta_vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    order = []
    visited = set()

    def sort_func(node):
        if node.unique_id in visited or node.is_constant():
            return
        
        visited.add(node.unique_id)

        for parent in node.parents:
            sort_func(parent)

        order.append(node)

    sort_func(variable)
    return order

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    order = topological_sort(variable)
    grads = {}
    grads[variable.unique_id] = deriv

    for var in reversed(order):
        if var.is_leaf():
            continue
            
        curr_deriv = grads.get(var.unique_id, 0.0)

        parent_deriv = var.chain_rule(curr_deriv)

        for parent, parent_grad in parent_deriv:
            if parent.is_leaf():
                parent.accumulate_derivative(parent_grad)
            else:
                if parent.unique_id not in grads:
                    grads[parent.unique_id] = 0.0
                grads[parent.unique_id] += parent_grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
