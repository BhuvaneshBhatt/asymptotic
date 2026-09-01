"""Documentation examples are executable contracts, not copied snippets."""

from examples.advanced_saddles_and_sums import main as advanced_saddle_example
from examples.certified_green import main as green_example
from examples.common_algebra import main as algebra_example
from examples.computational_complexity import main as complexity_example
from examples.ordinary_expansion import main as ordinary_example
from examples.probability_asymptotics import main as probability_example
from examples.singular_implicit import main as implicit_example


def test_documentation_examples_execute():
    for example in (
        ordinary_example,
        algebra_example,
        implicit_example,
        green_example,
        complexity_example,
        probability_example,
        advanced_saddle_example,
    ):
        example()
