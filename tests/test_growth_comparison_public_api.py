from asymptotic.context import AsymptoticGrowthComparison
from asymptotic.relative_growth import GrowthScaleComparison


def test_growth_comparison_public_api_is_unambiguous():
    assert AsymptoticGrowthComparison.LARGER.name == "LARGER"
    assert GrowthScaleComparison.GREATER.value == "greater"
