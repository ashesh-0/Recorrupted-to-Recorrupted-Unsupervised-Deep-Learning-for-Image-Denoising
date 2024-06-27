from typing import List

import numpy as np

class Enum:
    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def contains(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return True
        return False

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f'{cls.__name__}:{enum_type_str} doesnot exist.'


class DataSplitType(Enum):
    All = 0
    Train = 1
    Val = 2
    Test = 3


def split_in_half(s, e):
    n = e - s
    s1 = list(np.arange(n // 2))
    s2 = list(np.arange(n // 2, n))
    return [x + s for x in s1], [x + s for x in s2]


def adjust_for_imbalance_in_fraction_value(val: List[int], test: List[int], val_fraction: float, test_fraction: float,
                                           total_size: int):
    """
    here, val and test are divided almost equally. Here, we need to take into account their respective fractions
    and pick elements rendomly from one array and put in the other array.
    """
    if val_fraction == 0:
        test += val
        val = []
    elif test_fraction == 0:
        val += test
        test = []
    else:
        diff_fraction = test_fraction - val_fraction
        if diff_fraction > 0:
            imb_count = int(diff_fraction * total_size / 2)
            val = list(np.random.RandomState(seed=955).permutation(val))
            test += val[:imb_count]
            val = val[imb_count:]
        elif diff_fraction < 0:
            imb_count = int(-1 * diff_fraction * total_size / 2)
            test = list(np.random.RandomState(seed=955).permutation(test))
            val += test[:imb_count]
            test = test[imb_count:]
    return val, test


def get_datasplit_tuples(val_fraction: float, test_fraction: float, total_size: int, starting_test: bool = False):
    if starting_test:
        # test => val => train
        test = list(range(0, int(total_size * test_fraction)))
        last_test_idx = -1 if len(test) == 0 else test[-1]
        val = list(range(last_test_idx + 1, last_test_idx + 1 + int(total_size * val_fraction)))
        train = list(range(val[-1] + 1, total_size))
    else:
        # {test,val}=> train
        test_val_size = int((val_fraction + test_fraction) * total_size)
        train = list(range(test_val_size, total_size))

        if test_val_size == 0:
            test = []
            val = []
            return train, val, test

        # Split the test and validation in chunks.
        chunksize = max(1, min(3, test_val_size // 2))

        nchunks = test_val_size // chunksize

        test = []
        val = []
        s = 0
        for i in range(nchunks):
            if i % 2 == 0:
                val += list(np.arange(s, s + chunksize))
            else:
                test += list(np.arange(s, s + chunksize))
            s += chunksize

        if i % 2 == 0:
            test += list(np.arange(s, test_val_size))
        else:
            p1, p2 = split_in_half(s, test_val_size)
            test += p1
            val += p2

    val, test = adjust_for_imbalance_in_fraction_value(val, test, val_fraction, test_fraction, total_size)

    return train, val, test


if __name__ == '__main__':
    train, val, test = get_datasplit_tuples(0.8, 0.2, 20)
    print(train)
    print(val)
    print(test)

    train, val, test = get_datasplit_tuples(0.1, 0.1, 30, starting_test=True)
    print(train)
    print(val)
    print(test)
