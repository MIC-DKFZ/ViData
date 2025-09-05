import multiprocessing
import time
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import numpy as np
from tqdm import tqdm


def dummy_f(a=None, b=None, c=None, verbose=False, *args, **kwargs):
    """Dummy function for testing multiprocessing."""
    if verbose:
        print(a, b, c)
    time.sleep(3)
    return a + b + c, 0


def _apply_with_const(func: Callable[..., Any], const: dict[str, Any], args: dict[str, Any]) -> Any:
    """Top-level helper for multiprocessing compatibility."""
    return func(**args, **const)


def multiprocess_by_case(
    func: Callable[..., Any],
    iterable: list[dict[str, Any]],
    const: dict[str, Any] | None = None,
    p: int = 8,
    progressbar: bool = True,
    desc: str = "Processing",
) -> list[Any]:
    """
    Processes a list of dictionaries containing arguments for a function either by employing
    multiprocessing or by using a single-threaded approach, based on the specified number of worker processes.

    This function is useful for cases where each dictionary in the list represents a full set of arguments
    with which the function should be called. This allows for varied function calls with potentially different
    argument combinations in each call.

    Args:
        func: A callable to be applied to each set of arguments. This callable should accept
              keyword arguments only.
        iterable: A list of dictionaries, each representing a set of keyword arguments for `func`.
        const: An optional dictionary of additional constant keyword arguments that are included in
               every call to `func`. Defaults to None, meaning no constants are added.
        p: The number of worker processes to use for parallel processing. If set to 0 or None,
           the function performs sequential processing in the main process.
        progressbar: Specifies whether to display a progress bar during the execution to track
                     processing progress. Enabled by default.
        desc: A description text for the progress bar that provides context about the ongoing operation.

    Returns:
        A list of results obtained from applying `func` to each set of arguments in the list.
    """
    # const = const or {}
    # if p == 0 or p is None:
    #     # Sequential processing
    #     results = [
    #         func(**args, **const) for args in tqdm(iterable, desc=desc, disable=not progressbar)
    #     ]
    # else:
    #     # Parallel processing using multiprocessing
    #     with multiprocessing.Pool(processes=p) as pool:
    #         jobs = [pool.apply_async(func, kwds={**args, **const}) for args in iterable]
    #         results = [job.get() for job in tqdm(jobs, desc=desc, disable=not progressbar)]
    # return results
    const = const or {}

    if p == 0 or p is None:
        return [
            _apply_with_const(func, const, args)
            for args in tqdm(iterable, desc=desc, disable=not progressbar)
        ]

    with multiprocessing.Pool(processes=p) as pool:
        apply_func = partial(_apply_with_const, func, const)
        results = list(
            tqdm(
                pool.imap(apply_func, iterable),
                total=len(iterable),
                desc=desc,
                disable=not progressbar,
            )
        )
    return results


def multiprocess_iter(
    func: Callable[..., Any],
    iterables: dict[str, Iterable],
    const: dict[str, Any] | None = None,
    p: int = 8,
    progressbar: bool = True,
    desc: str = "Processing",
) -> list[Any]:
    """
    Processes data either by employing multiprocessing or by using a single-threaded approach,
    based on the specified number of worker processes.

    This function applies the callable `func` to every combination of arguments. These combinations are
    formed zip-wise from the values provided in `iterables`. Each key in the `iterables` dictionary
    corresponds to a keyword argument expected by `func`. Additionally, constant keyword arguments
    specified in `const` are included in each call to `func`. If `p` is set to a value greater than 0,
    the function utilizes multiprocessing with `p` worker processes to parallelize the execution. If `p`
    is 0 or None, the function executes sequentially in the main process.

    Args:
        func (Callable[..., Any]): A callable to be applied to each set of arguments. This callable
            should accept keyword arguments only.
        iterables (dict[str, Iterable]): A dictionary mapping from the names of `func`'s keyword arguments
            to iterables. Each iterable provides values for its corresponding argument in `func`.
        const (Optional[dict[str, Any]]): A dictionary of additional constant keyword arguments that are
            included in every call to `func`. Defaults to None, meaning no constants are added.
        p (int): The number of worker processes to use for parallel processing. If set to 0 or None,
            the function performs sequential processing in the main process.
        progressbar (bool): Specifies whether to display a progress bar during the execution to track
            processing progress. Enabled by default.
        desc (str): A description text for the progress bar that provides context about the ongoing operation.

    Returns:
        list[Any]: A list of results obtained from applying `func` to each combination of arguments
        derived from `iterables` and `const`.
    """
    keys = list(iterables.keys())
    values = list(zip(*iterables.values(), strict=False))  # Zips values by position
    arguments = [dict(zip(keys, v, strict=False)) for v in values]
    return multiprocess_by_case(
        func, arguments, const=const, p=p, progressbar=progressbar, desc=desc
    )


if __name__ == "__main__":
    a = np.ones(10) * 1
    b = np.ones(10) * 2
    c = np.ones(10) * 3

    res = multiprocess_iter(dummy_f, {"a": a, "b": b, "c": c}, {"verbose": True, "z": 3}, p=2)
    print(res)
    # res = multiprocess_list(dummy_f, {"a": a, "b": c, "c": c}, {"verbose": True, "z": 3}, p=8)
    # input = [
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    #     {"a": 1, "b": 2, "c": 3},
    # ]
    # res = multiprocess_list(dummy_f, input, {"verbose": True}, p=8)
