import inspect
import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, cast

from tqdm import tqdm


def _apply_with_const(
    func: Callable[..., Any], const: Mapping[str, Any], args: Mapping[str, Any]
) -> Any:
    """Top-level helper (pickle-safe) that merges `const` into each call."""
    return func(**args, **const)


def multiprocess_iter(
    func: Callable[..., Any],
    iterables: Mapping[str, Iterable] | Sequence[Iterable] | Sequence[Mapping[str, Any]],
    const: Mapping[str, Any] | None = None,
    p: int = 8,
    desc: str = "Processing",
    progressbar: bool = True,
    ordered: bool = True,
    chunksize: int | None = None,
) -> list[Any]:
    """
    Run `func` in parallel over inputs with a tqdm progress bar.

    iterables:
      - dict[str, Iterable]: columns -> zipped by position into kwargs rows (keys are arg names)
      - list[dict[str, Any]]: already kwargs rows -> forwarded as-is
      - list[Iterable]: columns -> arg names inferred from `func` (skips names present in `const`)
    const: kwargs added to every call
    p: number of worker processes (0/1 => run in main process)
    ordered: preserve input order (True) or yield as completed (False)
    chunksize: batching for speed (None => auto)
    """
    const = dict(const or {})

    # ---------- Normalize iterables into an iterator of kwargs rows ----------
    def _rows_from_dict(d: Mapping[str, Iterable]) -> Iterator[dict[str, Any]]:
        keys = list(d.keys())
        for vals in zip(*d.values(), strict=False):  # stops at shortest input
            yield dict(zip(keys, vals, strict=False))

    def _infer_keys(n_cols: int) -> list[str]:
        sig = inspect.signature(func)
        candidates = [
            name
            for name, prm in sig.parameters.items()
            if prm.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name not in const
        ]
        if n_cols > len(candidates):
            raise ValueError(
                f"Cannot infer {n_cols} argument names for {func.__name__}; "
                f"only {len(candidates)} available after excluding const keys {sorted(const.keys())}."
            )
        return candidates[:n_cols]

    def _len_or_none(x) -> int | None:
        return len(x) if hasattr(x, "__len__") else None

    arg_rows: Iterable[Mapping[str, Any]]
    total: int | None
    # Case A: dict of columns
    if isinstance(iterables, Mapping):
        arg_rows = _rows_from_dict(iterables)
        # Best-effort total: length of the first column if it supports len()
        try:
            first_col = next(iter(iterables.values()))
        except StopIteration:
            first_col = None
        total = _len_or_none(first_col) if first_col is not None else 0
        # Case B/C: sequence
    elif isinstance(iterables, Sequence) and len(iterables) > 0:
        first = iterables[0]

        # B: list of kwargs rows
        if isinstance(first, Mapping):
            # arg_rows = iter(iterables)  # already rows
            arg_rows = cast(Sequence[Mapping[str, Any]], iterables)  # already rows
            total = _len_or_none(iterables)

        # C: list of column iterables -> infer names, zip columns into rows
        else:
            keys = _infer_keys(len(iterables))

            def gen_rows() -> Iterator[dict[str, Any]]:
                for vals in zip(*iterables, strict=False):
                    yield dict(zip(keys, vals, strict=False))

            arg_rows = gen_rows()
            total = _len_or_none(iterables[0])
    else:
        return []
    # ---------- Execute sequentially or in parallel with tqdm ----------
    # apply_ = partial(_apply_with_const, func, const)
    apply_ = cast(Callable[[Mapping[str, Any]], Any], partial(_apply_with_const, func, const))
    # Sequential (debug-friendly)
    if p is None or p <= 1:
        return [
            apply_(row) for row in tqdm(arg_rows, total=total, desc=desc, disable=not progressbar)
        ]

    # Parallel
    if chunksize is None:
        chunksize = 1

    with mp.Pool(processes=p) as pool:
        imap = (
            pool.imap if ordered else pool.imap_unordered
        )  # predictable default; tune (e.g., 64/128) for many tiny tasks
        it = imap(apply_, arg_rows, chunksize)
        return list(tqdm(it, total=total, desc=desc, disable=not progressbar))


if __name__ == "__main__":
    import time

    def f(a, b, c, verbose=False):
        if verbose:
            print(a, b, c)
        time.sleep(a * 2)
        return a + b + c

    # 1) Dict of columns (zipped)
    input_args_1 = {"a": range(5), "b": range(5), "c": range(5)}
    print("Input", input_args_1)
    out = multiprocess_iter(f, input_args_1, const={"verbose": True}, p=2)
    print("Output", out)

    # 2) List of kwargs rows
    input_args_2 = [{"a": i, "b": i, "c": i} for i in range(5)]
    print("Input", input_args_2)
    out = multiprocess_iter(f, input_args_2, const={"verbose": True}, p=2)
    print("Output", out)

    # 3) List of columns (names inferred from f)
    input_args_3 = [range(5), range(5), range(5)]
    print("Input", input_args_3)
    out = multiprocess_iter(f, input_args_3, const={"verbose": True}, p=2)
    print("Output", out)
