# Recurrent Container class for Keras

The `RecurrentContainer` class unlocks the following scenarios that are currently not possible in [Keras](https://keras.io/):
- Multiple input and/or multiple output sequences.
- Feeding a constant input tensor to every timestep of the sequence.
- Arbitrary state tensors which can hold states for `Recurrent` layers, but also outputs from previous timesteps that can be fed back into the network.
- Recurrent network that takes a constant input and produces sequences.

# Recurrent Unit

This enables us to reuse existing `Recurrent` layers from Keras. Standard vanilla RNN units are defined by the following formula:
```
 h_t1 = sigmoid(h_t0 * R_kernel + x_t1 * X_kernel + bias)
 y_t1 = h_t1
```
Basically at timestep `t1` the RNN unit takes the *state* from the previous timestep `h_t0` along with the input *data* from the current timestep `x_t1` and produces the output *data* `y_t1` along with the *state* for the next timestep `h_t1` (which happen to be the same for vanilla RNN).
In Keras, the `SimpleRNN` unit applies the mentioned operation on whole sequences at once, i.e. it inputs a sequence and outputs a sequence. In order to build more complex models we may want to have direct access to state and output vectors. The `RecurrentUnit` class enables us to do this:
```python
y_t1, h_t1 = RecurrentUnit(SimpleRNN(10))(x_t1, states=h_t0)
```

# Building a model with a single RNN

This is the basic scenario which is already provided by `SimpleRNN`. We show here how to recreate it with `RecurrentContainer`:
```python
from keras.layers import SimpleRnn, Input
from keras.models import Model
from recurrentcontainer import RecurrentUnit, RecurrentContainer

x_in, h_in = Input(shape=(20,)), Input(shape=(10,))
x_out, h_out = RecurrentUnit(SimpleRNN(10))(x_in, states=h_in)

data_in = Input(shape=(None, 20))
l_rcontainer = RecurrentContainer(sequence_inputs=x_in, sequence_outputs=x_out, state_inputs=h_in, state_outputs=h_out)
data_out = l_rcontainer(sequences=data_in)

model = Model(inputs=data_in, outputs=data_out)
```
