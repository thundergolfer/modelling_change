import random
from typing import List, Optional, Dict, Set, Tuple, Union


def _italic_str(text: str) -> str:
    return f"\x1B[3m{text}\x1B[23m"


def _superscript_exp(n: str) -> str:
    return "".join(["⁰¹²³⁴⁵⁶⁷⁸⁹"[ord(c) - ord('0')] for c in str(n)])


class Variable:
    """
    A object representing a mathematical variable, for use in building expressions.

    Usage: `x = Variable("x")`
    """
    def __init__(self, var: str):
        if len(var) != 1 or (not var.isalpha()):
            raise ValueError("Variable must be single alphabetical character. eg. 'x'")
        self.var = var

    def __repr__(self):
        return _italic_str(self.var)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Variable):
            return self.var == other.var
        return False

    def __key(self):
        return self.var

    def __hash__(self):
        return hash(self.__key())


# An element of some set called a space. Here, that 'space' will be the domain of a multi-variable function.
Point = Dict[Variable, float]


class Expression:
    def diff(self, ref_var: Optional[Variable] = None) -> Optional["Expression"]:
        raise NotImplementedError

    def evaluate(self, point: Point) -> float:
        raise NotImplementedError


class ConstantExpression(Expression):
    """
    ConstantExpression is a single real-valued number.
    It cannot be parameterised and it's first-derivative is always 0 (None).
    """

    def __init__(self, real: float):
        super().__init__()
        self.real = real

    def diff(self, ref_var: Optional[Variable] = None) -> Optional[Expression]:
        return None

    def evaluate(self, point: Point) -> float:
        return self.real

    def __repr__(self):
        return str(self.real)


class PolynomialExpression(Expression):
    """
    An expression object that support evaluation and differentiation of single-variable polynomials.

    # TODO(Jonathon): Support multi-variable expressions. Eg. 4xy^2z^2
    """
    def __init__(
            self,
            variable: Variable,
            coefficient: float,
            exponent: int
    ):
        super().__init__()
        self.var = variable
        self.coefficient = coefficient
        self.exp = exponent

    def diff(self, ref_var: Optional[Variable] = None) -> Optional[Expression]:
        if ref_var and ref_var != self.var:
            return None
        if self.exp == 1:
            return ConstantExpression(real=self.coefficient)
        return PolynomialExpression(
            variable=self.var,
            coefficient=self.coefficient * self.exp,
            exponent=self.exp - 1,
        )

    def evaluate(self, point: Point) -> float:
        return (
                self.coefficient *
                point[self.var] ** self.exp
        )

    def __repr__(self):
        return f"{self.coefficient}{self.var}{_superscript_exp(str(self.exp))}"


GradientVector = Dict[Variable, Union["MultiVariableFunction", float]]


class MultiVariableFunction:
    """
    MultiVariableFunction support the composition of expressions by addition into a
    function of multiple real-valued variables.

    Partial differentiation with respect to a single variable is supported, as is
    evaluation at a Point, and gradient finding.
    """

    def __init__(self, variables: Set[Variable], expressions: List[Expression]):
        self.vars = variables
        self.expressions = expressions

    def gradient(self, point: Optional[Point] = None) -> GradientVector:
        grad_v: GradientVector = {}
        for v in self.vars:
            grad_v[v] = self.diff(ref_var=v)
        if point:
            return {
                var: f.evaluate(point)
                for var, f
                in grad_v.items()
            }
        else:
            return grad_v

    def diff(self, ref_var: Variable) -> "MultiVariableFunction":
        first_partial_derivatives: List[Expression] = []
        for expression in self.expressions:
            first_partial_diff = expression.diff(ref_var=ref_var)
            if first_partial_diff:
                first_partial_derivatives.append(first_partial_diff)
        return MultiVariableFunction(
            variables=self.vars,
            expressions=first_partial_derivatives,
        )

    def evaluate(self, point: Point) -> float:
        return sum(
            expression.evaluate(point)
            for expression
            in self.expressions
        )

    def __repr__(self):
        return " + ".join([str(e) for e in self.expressions])


def gradient_descent(
        gamma: float,
        max_iterations: int,
        f: MultiVariableFunction,
) -> Tuple[float, Point]:
    """
    Implements Gradient Descent (https://en.wikipedia.org/wiki/Gradient_descent) in pure-Python3.6+ with
    no external dependencies.

    :param gamma: 'step size', or 'learning rate'
    :param max_iterations: Maximum number of steps in descent process.
    :param f: A differentiable function off multiple real-valued variables.
    :return: A tuple of first a local minimum and second the point at which minimum is found.
    """
    if gamma <= 0:
        raise ValueError("gamma value must be a positive real number, γ∈ℝ+")

    a: Point = {}
    for v in f.vars:
        a[v] = random.randrange(4)
    for i in range(max_iterations):
        grad_a: GradientVector = f.gradient(a)
        # update estimate of minimum point
        a_next = {
            var: current - (gamma * grad_a[var])
            for var, current
            in a.items()
        }
        a = a_next
        if i % 10 == 0:
            print(f"Current min estimate: {a}")
    return f.evaluate(a), a


def main() -> None:
    print("hello world")
    x = Variable("x")
    y = Variable("y")
    exp = PolynomialExpression(
        variable=x,
        coefficient=2,
        exponent=4,
    )
    print(exp)
    print(exp.diff())

    test_f = MultiVariableFunction(
        variables={x, y},
        expressions=[
            PolynomialExpression(variable=x, coefficient=1, exponent=2),
            PolynomialExpression(variable=y, coefficient=1, exponent=2),
            PolynomialExpression(variable=x, coefficient=-2, exponent=1),
            PolynomialExpression(variable=y, coefficient=-6, exponent=1),
            ConstantExpression(real=14.0),
        ],
    )
    minimum_val, minimum_point = gradient_descent(
        gamma=0.1,
        max_iterations=5000,
        f=test_f,
    )
    print(f"Min Value: {minimum_val}")
    print(f"Min Location: {minimum_point}")

    # Test variable comparisons
    ##########################
    assert Variable("x") == Variable("x")
    assert Variable("x") != Variable("y")
    assert Variable("y") != Variable("x")
    assert Variable("y") != Variable("z")

    # Test gradient evaluations of Expressions
    ##########################################
    # ConstantExpressions
    assert ConstantExpression(real=0.0).diff() is None
    assert ConstantExpression(real=4.5).diff() is None
    # PolynomialExpression
    poly1 = PolynomialExpression(
        variable=Variable("x"),
        coefficient=2,
        exponent=4,
    )
    poly1_grad1 = poly1.diff()
    assert poly1_grad1.var == Variable("x")
    assert poly1_grad1.coefficient == 8
    assert poly1_grad1.exp == 3
    poly1_grad2 = poly1.diff(ref_var=Variable("y"))
    assert poly1_grad2 is None

    # Test function evaluation
    ##########################
    x = Variable("x")
    y = Variable("y")
    # f = 3x + y^2
    f1 = MultiVariableFunction(
        variables={x, y},
        expressions=[
            PolynomialExpression(variable=x, coefficient=3, exponent=1),
            PolynomialExpression(variable=y, coefficient=1, exponent=2),
        ],
    )
    assert f1.evaluate(point={x: 1.0, y: 1.0}) == 4
    assert f1.evaluate(point={x: 1.0, y: 2.0}) == 7
    # Test function gradient
    g = f1.gradient()
    assert str(g[x]) == "3"


if __name__ == "__main__":
    main()
