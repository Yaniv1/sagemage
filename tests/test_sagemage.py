"""Basic tests for sagemage package."""

import sagemage
import random

import pandas as pd


def test_package_version():
    """Test that package has a version attribute."""
    assert hasattr(sagemage, "__version__")
    assert isinstance(sagemage.__version__, str)


def test_package_import():
    """Test that sagemage can be imported."""
    import sagemage

    assert sagemage is not None


def test_dataset_get_chunks_max_chunks_number():
    """max_chunks=N keeps first N chunks."""
    ds = sagemage.Dataset(
        input_data=pd.DataFrame({"id": [1, 2, 3, 4, 5], "text": ["a", "b", "c", "d", "e"]}),
        id_column="id",
        columns=["text"],
        chunk_size=1,
    )

    chunks = ds.get_chunks(max_chunks=3)
    assert len(chunks) == 3
    assert [ck.split("_chunk_")[-1].split(".")[0] for ck, _ in chunks] == ["00000", "00001", "00002"]


def test_dataset_get_chunks_max_chunks_range_and_sample_order():
    """sample is applied after max_chunks restriction."""
    ds = sagemage.Dataset(
        input_data=pd.DataFrame({"id": [1, 2, 3, 4, 5], "text": ["a", "b", "c", "d", "e"]}),
        id_column="id",
        columns=["text"],
        chunk_size=1,
    )

    all_chunks = ds.get_chunks()
    first_three_keys = {ck for ck, _ in all_chunks[:3]}

    random.seed(123)
    sampled = ds.get_chunks(max_chunks=3, sample=2)

    sampled_keys = {ck for ck, _ in sampled}
    assert len(sampled) == 2
    assert sampled_keys.issubset(first_three_keys)


def test_dataset_get_chunks_no_restriction_values():
    """None/False/0/-1 for max_chunks and False for sample mean no restriction."""
    ds = sagemage.Dataset(
        input_data=pd.DataFrame({"id": [1, 2, 3], "text": ["a", "b", "c"]}),
        id_column="id",
        columns=["text"],
        chunk_size=1,
    )

    expected = len(ds.get_chunks())
    for mc in [None, False, 0, -1]:
        assert len(ds.get_chunks(max_chunks=mc, sample=False)) == expected
