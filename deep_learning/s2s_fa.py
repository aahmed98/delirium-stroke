import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator

# Configuration Parameters
window_sz = 60
num_units = 32
batch_sz = 64
epochs = 10

def createTFDataset(seq_data, id_list):
    """
    Converts sequence data and ID list into a TensorFlow dataset for training.
    
    Args:
        seq_data: List of arrays representing sequential data (days, timesteps, features).
        id_list: List of patient IDs corresponding to seq_data.
        
    Returns:
        tf_data: Normalized and imputed tensor of data.
        dataset: Batched TensorFlow dataset for training.
    """
    assert len(seq_data) == len(id_list), f"seq:{len(seq_data)}, ids:{len(id_list)}"

    tf_data = None
    last_hours = []
    first_hours = []
    prev_patient = -1

    for day, patient_id in zip(seq_data, id_list):
        if patient_id != prev_patient:
            if len(first_hours) == 0:
                first_hours.append(0)
                prev_patient = patient_id
                continue
            last_hours.append(len(tf_data) - 1 if tf_data is not None else 0)
            first_hours.append(len(tf_data) if tf_data is not None else 0)
            prev_patient = patient_id

        ts = TimeseriesGenerator(day, day, length=window_sz, stride=window_sz, shuffle=False)
        assert len(ts) == 1
        x, _ = ts[0]
        tf_data = x if tf_data is None else tf.concat([tf_data, x], axis=0)

    last_hours.append(len(tf_data) - 1)

    x_indices = [i for i in range(len(tf_data)) if i not in set(last_hours)]
    y_indices = [i for i in range(len(tf_data)) if i not in set(first_hours)]

    # Normalize data
    tf_data = (tf_data - tf.math.reduce_mean(tf_data, axis=0)) / tf.math.reduce_std(tf_data, axis=0)
    tf_data = tf.where(tf.math.is_nan(tf_data), tf.zeros_like(tf_data), tf_data)

    # Create dataset
    x_train = tf.gather(tf_data, x_indices)
    y_train = tf.gather(tf_data, y_indices)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(batch_sz, drop_remainder=True)

    return tf_data, dataset

class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(6)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

def get_pretrained_encoder():
    """
    Loads a pretrained encoder and decoder model with restored checkpoints.
    """
    encoder = Encoder(num_units, batch_sz)
    decoder = Decoder(num_units, batch_sz)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    checkpoint_dir = 'deep_learning/training_checkpoints'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    return encoder, num_units

loss_object = tf.keras.losses.MeanSquaredError()

def loss_function(real, pred):
    """
    Calculates mean squared error loss, masking zeroed-out values in the real target.
    """
    pred = tf.where(real != 0.0, pred, 0.0)
    loss_ = loss_object(real, pred)
    return loss_