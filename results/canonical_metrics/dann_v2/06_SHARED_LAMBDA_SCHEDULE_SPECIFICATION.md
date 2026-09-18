# Shared lambda schedule specification

The only schedule implementation is:

`lambda = lambda_max * (2 / (1 + exp(-10 * p)) - 1)`

`p = global_update_index / (50 * registered_steps_per_epoch - 1)`

The zero-based update index is local to the training run. `registered_steps_per_epoch` is fixed from the registered training population and batch size before the first update. The denominator always represents 50 epochs, including early-stopped development and shorter final refits. A1 and A2 consume the same value at every matched update; only sign differs.

Every run records each update value, mean value, maximum realized value, and cumulative absolute encoder-domain coefficient.

