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


## POMDP description

### state space (`pomdp_py.OOState`)
* Robot state (id, pose, objects_found, camera_direction)
* Object state (id, pose, time)

### action space
* Motion actions: MoveEast, MoveWest, MoveSouth, MoveNorth
* Look action (optional)
* Find action

### observation space
* Object observation: NULL or object pose

### transition_model
* Robot transition: deterministic
* Object transition: depends on object's motion policy

### observation_model
* See [Wandzel'19](https://ieeexplore.ieee.org/iel7/8780387/8793254/08793888.pdf?casa_token=AALwvAe4wJYAAAAA:KchwvsdTTl-cEftOo79XJxxLUlHwIGToQLIf8TdpEEkEa2Jp09tFq-HTEkAvjUL4Udw2NVizLw8)

### reward_model
* +/-100 for Find. +/-1 for Look and Move. The Move action receives additional penalty on distance moved (-1 per grid cell).

### policy_model
* If Look action is enabled, then Find can only be taken after Look is performed.
