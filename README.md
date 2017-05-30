# Recurrent Container class for Keras

The `RecurrentContainer` class unlocks the following scenarios that are currently not possible in Keras:
- Multiple input and/or multiple output sequences.
- Feeding a constant input tensor to every timestep of the sequence.
- Arbitrary state tensors which can hold states for `Recurrent` layers, but also outputs from previous timesteps that can be fed back into the network.
- Recurrent network that takes a constant input and produces sequences.
