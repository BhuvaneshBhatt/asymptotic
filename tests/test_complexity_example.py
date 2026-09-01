import pytest

from examples.computational_complexity import multiplication_pair_count


@pytest.mark.parametrize("size", [2, 4, 8])
def test_transseries_product_candidate_count_is_quadratic_for_equal_sizes(size):
    retained, products = multiplication_pair_count(size)
    assert retained == size
    assert products == size**2


def test_complexity_example_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="positive"):
        multiplication_pair_count(0)
