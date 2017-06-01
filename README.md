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

This is the basic scenario which is already provided in Keras by `SimpleRNN`:
```python
from keras.layers import SimpleRnn, Input
from keras.models import Model

data_in = Input(shape=(None, 20))
data_out = SimpleRNN(20)(data_in)

model = Model(inputs=data_in, outputs=data_out)
```
This model takes a 3D sequence input tensor and produces a 2D tensor that represents the output from the final timestep of the sequence. We show here how to recreate it with `RecurrentContainer`:
```python
from keras.layers import SimpleRnn, Input
from keras.models import Model
from recurrentcontainer import RecurrentUnit, RecurrentContainer

# Constructing the graph of the recurrent model.
x_in, h_in = Input(shape=(20,)), Input(shape=(10,))
x_out, h_out = RecurrentUnit(SimpleRNN(10))(x_in, states=h_in)

# Building a recurrent container for the graph.
data_in = Input(shape=(None, 20))
l_rcontainer = RecurrentContainer(
    sequence_inputs=x_in,
    sequence_outputs=x_out,
    state_inputs=h_in,
    constant_outputs=h_out)
data_out = l_rcontainer(sequences=data_in)

model = Model(inputs=data_in, outputs=data_out)
```
First we always need to construct the graph of the recurrent model. The `RecurrentContainer` is instantiated by providing input and output tensors of the graph. When the recurrent container is called, input tensors are grouped according to their semantics into three groups:
* `sequences`: 3D tensors of shape `(batch_size, timesteps, input_dim)` which are sliced over the `timesteps`. The slices are fed one by one in each timestep into corresponding 2D tensors of the recurrent model graph.
* `constants`: 2D tensors of shape `(batch_size, input_dim)` that will be fed directly into the corresponding tensors of the recurrent model graph. They are used when we want to feed the same input to all timesteps.
* `states`: 2D tensors of shape `(batch_size, num_units)` that are fed in the first timestep to initialize the state input tensors of the recurrent model graph.
