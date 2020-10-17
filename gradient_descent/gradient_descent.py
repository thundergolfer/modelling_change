import random
from typing import List, Optional, Mapping, Set, Union


def _italic_str(text: str) -> str:
    return f"\x1B[3m{text}\x1B[23m"


def _superscript_exp(n: str) -> str:
    return "".join(["⁰¹²³⁴⁵⁶⁷⁸⁹"[ord(c) - ord('0')] for c in str(n)])


# TODO(Jonathon): https://stackoverflow.com/a/13559470/4885590
class Variable:
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


Point = Mapping[Variable, float]


class Expression:
    """
    TODO
    """

    def __init__(self):
        pass

    def diff(self, ref_var: Optional[Variable] = None) -> Optional["Expression"]:
        raise NotImplementedError

    def evaluate(self, point: Point) -> float:
        raise NotImplementedError


class ConstantExpression(Expression):
    """

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
    # TODO(Jonathon): Multi-variable expression
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


GradientVector = Mapping[Variable, Union["MultiVariableFunction", float]]


class MultiVariableFunction:
    """
    TODO
    """

    def __init__(self, variables: Set[Variable], expressions: List[Expression]):
        self.vars = variables
        self.expressions = expressions

    def gradient(self, point: Optional[Point] = None) -> GradientVector:
        # TODO(Jonathon): implement
        if point:
            raise RuntimeError("Fuck fuck")
        grad_v: GradientVector = {}
        for v in self.vars:
            grad_v[v] = self.diff(ref_var=v)
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
) -> (float, Point):
    if gamma <= 0:
        raise ValueError("gamma value must be a positive real number, γ∈ℝ+")

    a: Point = {}
    for v in f.vars:
        a[v] = random.randrange(100)
    for i in range(max_iterations):
        grad_a = f.gradient(a)
    return 0, {}


def main() -> None:
    print("hello world")
    x = Variable("x")
    exp = PolynomialExpression(
        variable=x,
        coefficient=2,
        exponent=4,
    )
    print(exp)
    print(exp.diff())

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
    poly1_grad = PolynomialExpression(
        variable=Variable("x"),
        coefficient=2,
        exponent=4,
    ).diff()
    assert poly1_grad.var == Variable("x")
    assert poly1_grad.coefficient == 8
    assert poly1_grad.exp == 3

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
    print(g)



if __name__ == "__main__":
    main()
