import copy
import keras.backend as K
import numpy as np

from keras.engine import InputSpec
from keras.engine.topology import Layer, Container
from keras.layers import Input, Recurrent, Wrapper


class RecurrentUnit(Wrapper):
    """
    This wrapper can only be applied to layers derived from `Recurrent`. When called, it applies
    a the `step()` function of the wrapped layer to input tensors. The step function takes
    an input data tensor and an input state tensor and produces an output data tensor and
    an output state tensor.

    ```python
        t_data_in, t_state_in = Input((20,)), Input((10,))
        l_wrapped = RecurrentUnit(SimpleRNN(10))
        t_data_out, t_state_out = l_wrapped(t_data_in, states=t_state_in)
    ```
    """

    def __init__(self, layer, **kwargs):
        """
        Creates an instance of the `RecurrentUnit` as a wrapper around a given recurrent layer.

        # Arguments
            layer: A layer derived from `Recurrent` to be wrapped.
        """

        assert isinstance(layer, Recurrent)

        super(RecurrentUnit, self).__init__(layer, **kwargs)

        self.supports_masking = True
        self.layer.implementation = 1   # TODO: Maybe change.
        self.layer.return_sequences = False


    def build(self, input_shape):
        """
        Creates the layer weights.

        # Arguments
            input_shape: Keras tensor (future input to layer) or list/tuple of Keras tensors
                to reference for weight shape computations.
        """
        
        if not isinstance(input_shape, list):
            input_shape = [input_shape]

        # Build layer if not already built.
        if not self.layer.built:
            if self.num_inputs == 0:
                # If there are no inputs, we will feed in an all-zero input of dimension 1.
                child_input_shape = (input_shape[0][0], None, 1)
            else:
                child_input_shape = [(x[0], None) + x[1:] for x in input_shape[:self.num_inputs]]

            self.layer.build(child_input_shape)
            self.layer.built = True

        self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
        self.state_spec = self.layer.state_spec
        if not isinstance(self.state_spec, list):
            self.state_spec = [self.state_spec]

        # Call build of parent class.
        super(RecurrentUnit, self).build()


    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Assumes that the layer will be built to match that input shape provided.

        # Arguments
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per
                output tensor of the layer). Shape tuples can include `None` for free dimensions,
                instead of an integer.

        # Returns
            An output shape tuple.
        """

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        # Works because self.layer.return_sequences is set to False.
        child_input_shape = (input_shape[0], None) + input_shape[1:]
        data_output_shape = self.layer.compute_output_shape(child_input_shape)

        return [data_output_shape] + [x.shape for x in self.state_spec]


    def compute_mask(self, inputs, mask):
        """Computes an output mask tensor.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        """
        return [None] * (1 + len(self.state_spec))


    def reset_states(self, states_value=None):
        """
        Resets the states of the wrapped recurrent layer.
        """
        self.layer.reset_states(states_value)


    def call(self, inputs, num_inputs, training=None):
        """
        This is where the layer's logic lives. Calls the `step()` function of the wrapped
        recurrent layer.

        # Arguments
            inputs: List containing data and state input tensors.

        # Returns
            List containing data and state output tensors.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Separate inputs and states. Collect constants (the wrapped layer expects a 3D input).
        constants = self.layer.get_constants(K.expand_dims(inputs[0], 1))
        states = inputs[num_inputs:]
        inputs = inputs[:num_inputs]

        if num_inputs == 0:
            inputs = K.sum(K.zeros_like(states[0]), axis=1, keepdims=True)
        elif num_inputs == 1:
            inputs = inputs[0]

        output, output_states = self.layer.step(inputs, states + constants)

        # Properly set learning phase.
        if 0 < self.layer.dropout + self.layer.recurrent_dropout:
            output._uses_learning_phase = True

        return [output] + output_states


    def __call__(self, inputs, states=None, **kwargs):
        """
        Calls the wrapper layer with data and state inputs. The number of state tensors
        depends on the wraped layer (e.g. `SimpleRNN` and `GRU` have one while `LSTM` has two).

        All input tensors are passed to the `step()` function of the wrapped recurrent layer.

        # Arguments
            inputs: A single data input tensor of shape `(batch_size, input_dim)`.
            states: One or more state input tensors of shape `(batch_size, output_dim)`.

        # Returns
            A tuple where the first element is the data output tensor and the remaining elements
            are state output tensors. The shapes are the same as corresponding input tensors.
        """

        if inputs is None:
            inputs = []
        if not isinstance(inputs, list):
            inputs = [inputs]

        if 'num_inputs' in kwargs:
            self.num_inputs = kwargs['num_inputs']
            states = inputs[self.num_inputs:]
            inputs = inputs[:self.num_inputs]
        else:
            kwargs['num_inputs'] = len(inputs)
            self.num_inputs = kwargs['num_inputs']

        if states is None:
            raise ValueError('A RecurrentUnit must be passed input states when called.')
        if not isinstance(states, list):
            states = [states]

        # Perform the call. The outputs contain both the output and the states.
        self.input_spec = [self.input_spec] * len(inputs) + [None] * len(states)
        outputs = super(RecurrentUnit, self).__call__(inputs + states, **kwargs)

        return outputs


class RecurrentContainer(Recurrent):
    """
    Generalization of a `Recurrent` layer that enables the user to store an arbitrary layer graph
    with multiple input and output tensors and let it behave like a recurrent model.

    # Arguments

        sequence_inputs: Tensors that will be fed one 2D matrix in each timestep of the sequence.
        sequence_outputs: Tensors that will output one 2D matrix in each timestep of the sequence.
        constant_inputs: 2D tensors that are fed to the network and remain constant throughout
            the whole sequence.
        constant_outputs: 2D tensors that are outputted from the network and at the last
            timestep of the sequence.
        state_inputs: Tensors that will be fed with 2D values that were outputted by
            `state_outputs` in the previous timestep.
        state_outputs: Tensors that will output a 2D state value in each timestep that will
            be fed into `state_inputs` in the next timestep.
        rerutn_states: If set to `True` when a `RecurrentContainer` instance is called,
            it will also return state outputs, along with sequence and constant outputs.
        output_length: The length of the output. Should be specified only if there are no
            `sequence_inputs` in which case our network is basically supposed to produce a
            sequence based on a constant input and/or an initial state.

    # Input Shapes

    When called, the layer can be passed three types of input tensors:

        `sequences`: 3D tensors of shape `(batch_size, timesteps, input_dim)` which will
            be fed sliced over the `timesteps` dimension and fed in a sequence into corresponding
            tensors in `sequence_inputs` of shape `(batch_size, input_dim)`.
        `constants`: 2D tensors of shape `(batch_size, input_dim)` that will be fed directly to the
            tensors of the same shape defined in `constant_inputs`.
        `states`: 2D state tensors of shape `(batch_size, num_units)` that will be fed in the
            first timestep into the tensors defined in `state_inputs`. If omitted, the state
            tensors are initialized with all-zero matrices.
    
    Note that the sizes of the `batch_size` and `timesteps` dimensions should be the same for all
    input tensors, while the `input_dim` and `num_units` can be different among tensors.
    
    # Output Shapes

    When called the layer outputs a list of tensors that contains sequence outputs, followed by
    constant outputs and then optionally (if `return_states` was set to `True`) state outputs.

    """

    def __init__(self, sequence_inputs=[], sequence_outputs=[],
        constant_inputs=[], constant_outputs=[],
        state_inputs=[], state_outputs=[],
        return_states=False,
        output_length=None, **kwargs):

        super(RecurrentContainer, self).__init__(**kwargs)

        # Basic assertions that ensure the container is used as intended.
        assert len(state_inputs) == len(state_outputs)
        assert len(sequence_inputs + constant_inputs + state_inputs) > 0
        assert len(sequence_outputs + constant_outputs + state_outputs) > 0

        # Any input list could have been passed as a single element.
        if not isinstance(sequence_inputs, list):
            sequence_inputs = [sequence_inputs]
        if not isinstance(sequence_outputs, list):
            sequence_outputs = [sequence_outputs]
        if not isinstance(constant_inputs, list):
            constant_inputs = [constant_inputs]
        if not isinstance(constant_outputs, list):
            constant_outputs = [constant_outputs]
        if not isinstance(state_inputs, list):
            state_inputs = [state_inputs]
        if not isinstance(state_outputs, list):
            state_outputs = [state_outputs]

        self.sequence_inputs = sequence_inputs
        self.sequence_outputs = sequence_outputs
        self.constant_inputs = constant_inputs
        self.constant_outputs = constant_outputs
        self.state_inputs = state_inputs
        self.state_outputs = state_outputs

        self.return_states = return_states
        self.input_spec = None
        self.batch_size = None        # Size of the batch samples dimension.
        self.input_length = None      # Size of the timesteps dimension.
        self.output_length = output_length

        # Output length can only be specified if there are no sequence inputs.
        if len(sequence_inputs) > 0 and output_length is not None:
            raise ValueError('The "output_length" argument can only be specified if '
                'there are no sequence inputs.')
        elif len(sequence_inputs) == 0 and output_length is None:
            raise ValueError('If there are no sequence inputs, the output_length '
                'must be specified.')

        # Ensure tensors have the right dimensionality.
        assert all([K.ndim(x) == 2 for x in sequence_inputs + sequence_outputs +
            constant_inputs + constant_outputs + state_inputs + state_outputs])

        # Make sure we are returning states if there are no sequence or constant outputs.
        assert len(sequence_outputs + constant_outputs) > 0 or self.return_states

        # There cannot be tensors in multiple inputs lists.
        inputs = sequence_inputs + constant_inputs + state_inputs
        if len(inputs) > len(set(inputs)):
            raise ValueError('An input tensor cannot be in multiple lists at the same time.')

        # Output tensors can appear in multiple lists. Therefore we need to keep track of them.
        outputs = list(set(sequence_outputs) | set(constant_outputs) | set(state_outputs))
        self.sequence_outputs_indices = [outputs.index(x) for x in sequence_outputs]
        self.constant_outputs_indices = [outputs.index(x) for x in constant_outputs]
        self.state_outputs_indices = [outputs.index(x) for x in state_outputs]

        # Finally, construct the container.
        self.container = Container(inputs, outputs)

    
    def get_initial_states(self, inputs):
        """
        Returns a list of zero vectors that can be used to initialize the states.
        """

        if isinstance(inputs, list):
            inputs = inputs[0]

        # build an all-zero tensor of shape (samples, state_dim)
        state = K.zeros_like(inputs)  # (samples, *)
        state = K.sum(state, axis=tuple(range(1, K.ndim(inputs))))  # (samples,)
        state = K.expand_dims(state)  # (samples, 1)
        state_dims = [K.int_shape(x)[-1] for x in self.state_inputs]
        states = [K.tile(state, [1, x]) for x in state_dims] # (samples, state_dim)

        return states


    @property
    def input_spec(self):
        """
        Computes the input spec of the layer based on all input tensors that were defined.
        """

        # Trick to enable us to specify an artificial input spec.
        if hasattr(self, '_input_spec') and self._input_spec is not None:
            return self._input_spec

        # Collect the input spec from the container.
        input_spec = [InputSpec(shape=x) for x in self.container.internal_input_shapes]
        if not isinstance(input_spec, list):
            input_spec = [input_spec]

        # Add timesteps as the second dimension for sequence inputs.
        for i in range(len(self.sequence_inputs)):
            input_spec[i].shape = \
                input_spec[i].shape[:1] + (self.input_length,) + input_spec[i].shape[1:]
            input_spec[i].ndim += 1

        # Adjust the batch size dimension.
        for i in range(len(input_spec)):
            input_spec[i].shape = (self.batch_size,) + input_spec[i].shape[1:]

        return input_spec


    @input_spec.setter
    def input_spec(self, value):
        self._input_spec = value


    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Assumes that the layer will be built to match that input shape provided.

        # Arguments
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per
                output tensor of the layer). Shape tuples can include `None` for free dimensions,
                instead of an integer.

        # Returns
            An output shape tuple.
        """

        timesteps = self.output_length or self.input_length
        output_shape = []

        output_shape.extend([(self.batch_size, timesteps, K.int_shape(x)[1]) \
            for x in self.sequence_outputs])
        output_shape.extend([(self.batch_size,) + K.int_shape(x)[1:] \
            for x in self.constant_outputs])
        if self.return_states:
            output_shape.extend([(self.batch_size, K.int_shape(x)[1]) \
                for x in self.state_outputs])

        if len(output_shape) == 1:
            output_shape = output_shape[0]

        return output_shape


    def compute_mask(self, inputs, mask):
        """Computes an output mask tensor.

        # Arguments
            inputs: Tensor or list of tensors.
            mask: Tensor or list of tensors.

        # Returns
            None or a tensor (or list of tensors,
                one per output tensor of the layer).
        """

        if not isinstance(mask, list):
            mask = [mask]
        masks = [K.expand_dims(m, 0) for m in mask if m is not None]
        if len(masks) > 0:
            mask = K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)
        else:
            mask = None

        masks = [mask for _ in self.sequence_outputs]
        masks += [None for _ in self.constant_outputs]
        if self.return_states:
            masks += [None for _ in self.state_outputs]

        return masks


    def build(self, input_shape):
        """
        Creates the layer weights.

        # Arguments
            input_shape: Keras tensor (future input to layer) or list/tuple of Keras tensors
                to reference for weight shape computations.
        """

        if not isinstance(input_shape, list):
            input_shape = [input_shape]

        # Extract batch size from input shapes.
        batch_size = list(set([x[0] for x in input_shape if x[0] is not None]))
        if len(batch_size) > 1:
            raise ValueError('All input tensors should have the same batch size. ' + \
                'Received: [' + ', '.join(batch_size) + ']')
        self.batch_size = batch_size[0] if len(batch_size) == 1 else None

        # Extract number of timesteps from sequential inputs.
        if len(self.sequence_inputs) > 0:
            input_length = set([input_shape[i][1] for i in range(len(self.sequence_inputs))])
            input_length = list(input_length - set([None]))
            if len(input_length) > 1:
                raise ValueError('Shapes of sequence inputs should all have the same ' + \
                    'time dimension. Received: [' + ', '.join(input_length) + ']')
            self.input_length = input_length[0] if len(input_length) == 1 else None

        self.states = [None] * len(self.state_inputs)
        if self.stateful:
            self.reset_states()

        self.built = True


    def reset_states(self, states_value=None):
        """
        Resets the states of the recurrent container. Can only be called if
        the container is stateful.

        # Arguments
            state_values: List of numpy matrices with values to assign to state tensors.
        """

        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called and thus has no states.')

        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a recurrent container is stateful, it needs to know '
                             'its batch size. Specify the batch size of your input tensors: \n'
                             '- If using a Sequential model, specify the batch size by passing '
                             'a `batch_input_shape` argument to your first layer.\n'
                             '- If using the functional API, specify the time dimension '
                             'by passing a `batch_shape` argument to your Input layer.')
        if states_value is not None:
            if not isinstance(states_value, (list, tuple)):
                states_value = [states_value]
            if len(states_value) != len(self.states):
                raise ValueError('The layer has ' + str(len(self.states)) +
                                 ' states, but the `states_value` '
                                 'argument passed '
                                 'only has ' + str(len(states_value)) +
                                 ' entries')
        state_dims = [K.int_shape(x)[-1] for x in self.state_inputs]
        if self.states[0] is None:
            self.states = [K.zeros((batch_size, x)) for x in state_dims]
            if not states_value:
                return
        for i, state in enumerate(self.states):
            if states_value:
                value = states_value[i]
                if value.shape != (batch_size, state_dims[i]):
                    raise ValueError(
                        'Expected state #' + str(i) +
                        ' to have shape ' + str((batch_size, state_dims[i])) +
                        ' but got array with shape ' + str(value.shape))
            else:
                value = np.zeros((batch_size, state_dims[i]))
            K.set_value(state, value)


    def step(self, inputs, states):
        """
        Step function of the recurrent container.
        
        # Arguments
            input: tensor with shape `(samples, ...)` (no time dimension),
                representing input for the batch of samples at a certain
                time step.
            states: list of tensors.

        # Returns:
                output: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        """

        # Split the output tensors into corresponding groups.
        dims = [K.int_shape(x)[-1] for x in self.sequence_inputs]
        ranges = [(sum(dims[:i]), sum(dims[:i]) + dims[i]) for i in range(len(dims))]
        sequence_inputs = [inputs[:,x[0]:x[1]] for x in ranges]

        # Extract states and constants.
        states = list(states)
        state_inputs, constant_inputs = \
            states[:len(self.state_inputs)], states[len(self.state_inputs):]

        # Call the container and obtain outputs.
        all_outputs = self.container(sequence_inputs + constant_inputs + state_inputs)
        if not isinstance(all_outputs, list):
            all_outputs = [all_outputs]

        # Group output tensors by their indices.
        sequence_outputs = [all_outputs[i] for i in self.sequence_outputs_indices]
        constant_outputs = [all_outputs[i] for i in self.constant_outputs_indices]
        state_outputs = [all_outputs[i] for i in self.state_outputs_indices]

        # Build outputs by concatenating sequence and constant outputs.
        outputs = sequence_outputs + constant_outputs
        if len(outputs) > 1:
            outputs = K.concatenate(outputs)
        elif len(outputs) == 1:
            outputs = outputs[0]
        else:
            ndim = K.ndim(all_outputs[0])
            outputs = K.sum(K.zeros_like(all_outputs[0]), axis=tuple(range(1, ndim)))
            outputs = K.tile(K.expand_dims(outputs), (1, 0))

        return outputs, state_outputs


    def __call__(self, sequences=[], constants=[], states=[], **kwargs):
        """
        Calls this container with given input tensors.
        """
        
        # Ensure all input arguments are lists.
        if not isinstance(sequences, list):
            sequences = [sequences]
        if not isinstance(constants, list):
            constants = [constants]
        if not isinstance(states, list):
            states = [states]

        # This is used only for get_config and from_config.
        if 'num_sequences' in kwargs and 'num_constants' in kwargs and 'num_states' in kwargs:
            num_sequences = kwargs['num_sequences']
            num_constants = kwargs['num_constants']
            num_states = kwargs['num_states']
            inputs = sequences
            sequences, inputs = inputs[:num_sequences], inputs[num_sequences:]
            constants, inputs = inputs[:num_constants], inputs[num_constants:]
            states, inputs = inputs[:num_states], inputs[num_states:]
        else:
            kwargs['num_sequences'] = len(sequences)
            kwargs['num_constants'] = len(constants)
            kwargs['num_states'] = len(states)

        # Ensure sequences and constants are all present.
        assert len(sequences) == len(self.sequence_inputs)
        assert len(constants) == len(self.constant_inputs)

        # Ensure tensors have the right dimensionality.
        assert all([K.ndim(x) == 3 for x in sequences])
        assert all([K.ndim(x) == 2 for x in constants + states])

        # Build the states list. Replace None elements with default initial states.
        if self.stateful:
            if len(states) > 0:
                raise ValueError('Stateful recurrent layers cannot accept initial states.')
            initial_states = self.states
        else:
            inputs = [x for x in sequences + constants + states if x is not None]
            initial_states = self.get_initial_states(inputs)

        if len(states) == 0:
            states = initial_states
        else:
            for i in range(len(initial_states)):
                if states[i] is None:
                    states[i] = initial_states[i]

        # We do a trick to allow non-Keras tensors to be passed as inputs.
        inputs = sequences + constants + states
        is_keras_input = [K.is_keras_tensor(x) for x in inputs]
        non_keras_tensors = [None if is_keras_input[i] else inputs[i] for i in range(len(inputs))]
        inputs = [inputs[i] for i in range(len(inputs)) if is_keras_input[i]]
        self._input_spec = self.input_spec
        self._input_spec = [self._input_spec[i] \
            for i in range(len(self._input_spec)) if is_keras_input[i]]

        # Now inputs contains only Keras tensors, self.input_spec contains their specs and
        # non_keras_tensors is a list of non-Keras tensors with None values inserted in places
        # where Keras tensors were found in the original inputs.
        kwargs['non_keras_tensors'] = non_keras_tensors

        # Call the layer. Keras tensors are stored in inputs, the reast are in non_keras_tensors.
        outputs = Layer.__call__(self, inputs, **kwargs)

        # Reset the input spec.
        self._input_spec = None

        return outputs


    def call(self, inputs, mask=None, non_keras_tensors=None, training=None,
        num_sequences=None, num_constants=None, num_states=None):

        keras_tensor_inputs = inputs
        inputs = copy.copy(inputs)
        input_length = self.input_length or self.output_length

        # First reassemble inputs that were not Keras tensors.
        if non_keras_tensors is not None:
            assert isinstance(non_keras_tensors, list)
            non_keras_tensors = copy.copy(non_keras_tensors)
            for i in range(len(non_keras_tensors)):
                if non_keras_tensors[i] is None:
                    non_keras_tensors[i] = inputs.pop(0)
            inputs = non_keras_tensors
        assert len(inputs) > 0

        # Now split the inputs into separate groups.
        num_sequences = num_sequences or len(self.sequence_inputs)
        num_constants = num_constants or len(self.constant_inputs)
        num_states = num_states or len(self.state_inputs)
        sequence_inputs, inputs = inputs[:num_sequences], inputs[num_sequences:]
        constant_inputs, inputs = inputs[:num_constants], inputs[num_constants:]
        state_inputs, inputs = inputs[:num_states], inputs[num_states:]
        assert len(inputs) == 0

        # Assemble a list of inputs (sequential), states and constants. Since K.rnn() takes
        # a single 3D tensor as input, we concatenate all inputs along the last axis.
        # We also concatenate the readout mask.
        if len(sequence_inputs) > 1:
            sequence_inputs = K.concatenate(sequence_inputs)
        elif len(sequence_inputs) == 1:
            sequence_inputs = sequence_inputs[0]
        else:
            # If there are no sequential inputs, we will feed in a zero tensor with
            # shape (num_samples, num_timesteps, 0)
            sequence_inputs = K.zeros_like((constant_inputs + state_inputs)[0])
            sequence_inputs = K.sum(sequence_inputs, axis=tuple(range(1, K.ndim(sequence_inputs))))
            sequence_inputs = K.tile(K.expand_dims(sequence_inputs), (1, input_length))
            sequence_inputs = K.tile(K.expand_dims(sequence_inputs), (1, 1, 0))
        
        # Some sanity checks for unrolling.
        if self.unroll and input_length is None:
            raise ValueError('Cannot unroll a recurrent model if the time dimension is undefined.\n'
                             '- If using a Sequential model, specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` argument to your first '
                             'layer. If your first layer is an Embedding, you can also use the '
                             '`input_length` argument.\n'
                             '- If using the functional API, specify the time dimension by '
                             'passing a `shape` or `batch_shape` argument to your Input layer.')

        # Perform the rnn call.
        last_output, outputs, states = K.rnn(self.step,
                                             sequence_inputs,
                                             state_inputs,
                                             go_backwards = self.go_backwards,
                                             mask = mask,
                                             constants = constant_inputs,
                                             unroll = self.unroll,
                                             input_length = input_length)

        # Split the output tensors into corresponding groups.
        dims = [K.int_shape(x)[-1] for x in self.sequence_outputs + self.constant_outputs]
        ranges = [(sum(dims[:i]), sum(dims[:i]) + dims[i]) for i in range(len(dims))]
        sequence_outputs = [outputs[:,:,x[0]:x[1]] for x in ranges[:len(self.sequence_outputs)]]
        if len(self.sequence_outputs) > 0:
            constant_outputs = [outputs[:,-1,x[0]:x[1]] \
                for x in ranges[len(self.sequence_outputs):]]
        else:
            constant_outputs = [last_output[:,x[0]:x[1]] \
                for x in ranges[len(self.sequence_outputs):]]
        states = list(states)

        # Apply updates if stateful.
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, keras_tensor_inputs)

        # Properly set learning phase
        if self.container.uses_learning_phase:
            for x in sequence_outputs + constant_outputs + states:
                x._uses_learning_phase = True

        # Combine all output tensors and return them as a tuple.
        outputs = sequence_outputs + constant_outputs
        if self.return_states:
            outputs += states
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs
        

    @property
    def trainable_weights(self):
        return self.container.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.container.non_trainable_weights

    def get_weights(self):
        return self.container.get_weights()

    def set_weights(self, weights):
        self.container.set_weights(weights)

    def get_config(self):

        lengths = [len(self.sequence_inputs), len(self.constant_inputs), len(self.state_inputs)]
        indices = [range(sum(lengths[:i]), sum(lengths[:i+1])) for i in range(len(lengths))]

        config = {
            'container' : self.container.get_config(),
            'sequence_inputs_indices' : indices[0],
            'constant_inputs_indices' : indices[1],
            'state_inputs_indices' : indices[2],
            'sequence_outputs_indices' : self.sequence_outputs_indices,
            'constant_outputs_indices' : self.constant_outputs_indices,
            'state_outputs_indices' : self.state_outputs_indices,
            'return_states' : self.return_states,
            'output_length' : self.output_length
        }

        base_config = super(RecurrentContainer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        
        container = Container.from_config(config['container'])
        sequence_inputs = [container.inputs[i] for i in config['sequence_inputs_indices']]
        constant_inputs = [container.inputs[i] for i in config['constant_inputs_indices']]
        state_inputs = [container.inputs[i] for i in config['state_inputs_indices']]
        sequence_outputs = [container.outputs[i] for i in config['sequence_outputs_indices']]
        constant_outputs = [container.outputs[i] for i in config['constant_outputs_indices']]
        state_outputs = [container.outputs[i] for i in config['state_outputs_indices']]
        return_states = config['return_states']
        output_length = config['output_length']

        kwargs = {}
        for k in config:
            if k not in ['container', 'sequence_inputs_indices', 'constant_inputs_indices',
                'state_inputs_indices', 'sequence_outputs_indices', 'constant_outputs_indices',
                'state_outputs_indices', 'return_states', 'output_length']:
                kwargs[k] = config[k]

        return RecurrentContainer(sequence_inputs=sequence_inputs,
            sequence_outputs=sequence_outputs,
            constant_inputs=constant_inputs,
            constant_outputs=constant_outputs,
            state_inputs=state_inputs,
            state_outputs=state_outputs,
            return_states=return_states,
            output_length=output_length,
            **kwargs)
