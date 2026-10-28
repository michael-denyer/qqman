import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from qqman import qqman  # noqa: E402


def make_assoc(chroms, per_chrom=200):
    rng = np.random.default_rng(0)
    n = per_chrom * len(chroms)
    return pd.DataFrame(
        {
            "CHR": np.repeat(chroms, per_chrom),
            "BP": np.tile(np.arange(per_chrom) * 100, len(chroms)),
            "P": rng.uniform(1e-9, 1, n),
            "SNP": [f"rs{i}" for i in range(n)],
        }
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_manhattan_leaves_input_dataframe_unchanged():
    df = make_assoc([1, 2])
    before = df.copy()
    fig, ax = plt.subplots()

    qqman.manhattan(df, ax=ax)

    pd.testing.assert_frame_equal(df, before)
