# dynamic-object-search
Dynamic object search POMDP implemented using [pomdp_py](https://github.com/h2r/pomdp-py).


## How to use this package

### Running tests

The logic of a game (i.e. a run of the dynamic object search POMDP) is defined in
```
dynamic-object-search/dynamic_mos/experiments/runner.py
```
which contains `DynamicMosTrial` that has a `solve()` classmethod.

If all you want is to quickly run a single game, then use the `test.py`
under `dynamic-object-search/dynamic_mos/`. Modify this file however you want. Right now,
the `test_single` function calls the `unittest` function in `runner.py`
which builds an OOPOMDP problem and `solve()`s it.
