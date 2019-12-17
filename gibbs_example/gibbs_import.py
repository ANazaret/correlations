import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from scipy.special import gammaln

from functools import lru_cache


@lru_cache(1048576)
def log_fact(k):
    if k >= 1048570:
        raise Exception("Value too big for the cache system")
    if k < 0:
        raise ValueError
    if k <= 1:
        return 1

    return log_fact(k - 1) + np.log(k)


for i in range(1, 1048570):
    log_fact(i)