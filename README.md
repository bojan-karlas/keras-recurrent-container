# Recurrent Container class for Keras

The `RecurrentContainer` class unlocks the following scenarios that are currently not possible in [Keras](https://keras.io/):
- Multiple input and/or multiple output sequences.
- Feeding a constant input tensor to every timestep of the sequence.
- Arbitrary state tensors which can hold states for `Recurrent` layers, but also outputs from previous timesteps that can be fed back into the network.
- Recurrent network that takes a constant input and produces sequences.

# Recurrent Unit

This enables us to reuse existing `Recurrent` layers from Keras. Standard vanilla RNN units are defined by the following formula:
```
 h_t1 = sigmoid(h_t0 * R_kernel + x_t0 * X_kernel + bias)
 y_t1 = h_t1
```
