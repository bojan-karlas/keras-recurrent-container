import numpy as np
import keras.backend as K
import pytest
import sys

from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, Concatenate
from keras.models import Model
from keras.utils.test_utils import keras_test
from keras.utils import custom_object_scope

from recurrentcontainer import RecurrentUnit, RecurrentContainer

num_samples = 32
num_timesteps = 50
num_features = 6
num_units = 10
unit_types = {'rnn' : SimpleRNN, 'lstm' : LSTM, 'gru' : GRU}


def generate_data(shape, delay=0, invert=False):
    assert len(shape) >= 2
    shape = shape[:1] + (shape[1] + delay,) + shape[2:]
    x = np.random.randint(0, 2, shape).astype(float)
    y = x[:,:-delay,:] if len(shape) > 2 and delay > 0 else x
    x = x[:,delay:,:] if len(shape) > 2 and delay > 0 else x

    if invert:
        y = 1 - y

    return x, y


@keras_test
@pytest.mark.parametrize('unit,num_states', [('rnn', 1), ('gru', 1), ('lstm', 2)])
def test_recurrent_unit(unit, num_states):

    # Define input tensors for data and states.
    t_input_data = Input(batch_shape=(None, num_features))
    t_input_states = [Input(batch_shape=(None, num_units)) for _ in range(num_states)]

    # Construct the recurrent layer by wrapping RecurrentUnit around a given Recurrent class.
    l_recurrent = RecurrentUnit(unit_types[unit](num_units))
    t_outputs = l_recurrent(t_input_data, states=t_input_states)

    # Build a model.
    model = Model(inputs=[t_input_data] + t_input_states, outputs=t_outputs)
    model.compile(loss='mean_squared_error', optimizer = 'adam')

    # Construct data arrays.
    x, y = generate_data((num_samples, max(num_units, num_features)), invert=True)
    x, y = x[:, :num_features], x[:, :num_units]
    x_s, y_s = generate_data((num_samples, num_units), invert=True)
    
    # Fit the model.
    model.fit([x] + [x_s] * num_states, [y] + [y_s] * num_states, epochs=1)


@keras_test
@pytest.mark.parametrize('unit,num_states', [('rnn', 1), ('lstm', 2)])
@pytest.mark.parametrize('num_seq_inputs', [0,1,2])
@pytest.mark.parametrize('num_seq_outputs', [0,1,2])
@pytest.mark.parametrize('num_con_inputs', [0,1,2])
@pytest.mark.parametrize('num_con_outputs', [0,1,2])
@pytest.mark.parametrize('num_readouts', [0,1,2])
def test_recurrent_container(unit, num_states, num_seq_inputs, num_seq_outputs,
    num_con_inputs, num_con_outputs, num_readouts):

    # Build the recurrent container.

    t_seq_inputs = [Input(batch_shape=(None, num_features)) for _ in range(num_seq_inputs)]
    t_con_inputs = [Input(batch_shape=(None, num_features)) for _ in range(num_con_inputs)]
    t_ro_inputs = [Input(batch_shape=(None, num_features)) for _ in range(num_readouts)]
    t_state_inputs = [Input(batch_shape=(None, num_units)) for _ in range(num_states)]

    l_recurrent = RecurrentUnit(unit_types[unit](num_units, recurrent_dropout=0.5))
    t_inputs = t_seq_inputs + t_con_inputs + t_ro_inputs
    output_length = None
    if len(t_inputs) > 1:
        t_inputs = Concatenate()(t_inputs)
    elif len(t_inputs) == 0:
        t_inputs = None
    t_outputs = l_recurrent(t_inputs, states=t_state_inputs)
    t_data_output, t_state_outputs = t_outputs[0], t_outputs[1:]

    t_seq_outputs = [Dense(num_features)(t_data_output) for _ in range(num_seq_outputs)]
    t_con_outputs = [Dense(num_features)(t_data_output) for _ in range(num_con_outputs)]
    t_ro_outputs = [Dense(num_features)(t_data_output) for _ in range(num_readouts)]

    output_length = None if num_seq_inputs > 0 else num_timesteps

    l_container = RecurrentContainer(sequence_inputs=t_seq_inputs, sequence_outputs=t_seq_outputs,
        constant_inputs=t_con_inputs, constant_outputs=t_con_outputs,
        state_inputs=t_state_inputs + t_ro_inputs, state_outputs=t_state_outputs + t_ro_outputs,
        return_states=True, output_length=output_length)

    # Build the model.
    t_m_seq_inputs = [Input(batch_shape=(None, None, num_features)) for _ in range(num_seq_inputs)]
    t_m_con_inputs = [Input(batch_shape=(None, num_features)) for _ in range(num_con_inputs)]
    t_m_ro_inputs = [Input(batch_shape=(None, num_features)) for _ in range(num_readouts)]
    t_m_st_inputs = [Input(batch_shape=(None, num_units)) for _ in range(num_states)]
    t_m_outputs = l_container(sequences=t_m_seq_inputs, constants=t_m_con_inputs,
        states=t_m_st_inputs + t_m_ro_inputs)

    if not isinstance(t_m_outputs, list):
        t_m_outputs = [t_m_outputs]

    model = Model(inputs=t_m_seq_inputs + t_m_con_inputs + t_m_st_inputs + t_m_ro_inputs,
        outputs=t_m_outputs)

    # Convert model to config and back.
    config = model.get_config()
    custom_objects = {'RecurrentContainer' : RecurrentContainer, 'RecurrentUnit' : RecurrentUnit}
    with custom_object_scope(custom_objects):
        model = Model.from_config(config)

    model.compile(loss='mean_squared_error', optimizer = 'adam')

    # Build training data and fit model.
    x_seq, y_seq = generate_data((num_samples, num_timesteps, num_features), delay=1)
    x_con, y_con = generate_data((num_samples, num_features), invert=True)
    x_ro, y_ro = generate_data((num_samples, num_features), invert=True)
    x_st, y_st = generate_data((num_samples, num_units), invert=True)

    #tb = TensorBoard(log_dir='./logs/pytest', write_graph=True)

    x = [x_seq] * num_seq_inputs + [x_con] * num_con_inputs + \
        [x_st] * num_states + [x_ro] * num_readouts
    y = [y_seq] * num_seq_outputs + [x_con] * num_con_outputs + \
        [y_st] * num_states + [y_ro] * num_readouts
    model.fit(x, y, epochs=1)

    # Extract and apply weights.
    weights = model.get_weights()
    model.set_weights(weights)

    if K.backend() == 'tensorflow':
        K.clear_session()


if __name__ == '__main__':
    pytest.main(sys.argv)
