import tensorflow as tf
import math

class TdLSTM_error:
    def __init__(self, input_dim, output_dim, output_dim2, output_dim3, hidden_dim, hidden_dim2, hidden_dim3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim2 = output_dim2
        self.output_dim3 = output_dim3
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3

        self.initialize_weights_and_biases()

        # Placeholder for input, time, and dropout
        self.input = tf.placeholder('float', shape=[None, None, self.input_dim])
        self.time = tf.placeholder('float', shape=[None, None])
        self.keep_prob = tf.placeholder(tf.float32)

    def initialize_weights_and_biases(self):
        """Initialize weights and biases for the T-LSTM autoencoder layers."""
        # Layer 1 Encoder Weights
        self.Wq_enc = self.init_weights(3, self.hidden_dim, 'Input_Time_weight_enc')
        self.Wi_enc = self.init_weights(self.input_dim, self.hidden_dim, 'Input_Hidden_weight_enc')
        self.Ui_enc = self.init_weights(self.hidden_dim, self.hidden_dim, 'Input_State_weight_enc')
        self.bi_enc = self.init_bias(self.hidden_dim, 'Input_Hidden_bias_enc')

        # Layer 2 Encoder Weights
        self.Wq_enc2 = self.init_weights(3, self.hidden_dim2, 'Input_Time_weight_enc2')
        self.Wi_enc2 = self.init_weights(self.output_dim, self.hidden_dim2, 'Input_Hidden_weight_enc2')
        self.Ui_enc2 = self.init_weights(self.hidden_dim2, self.hidden_dim2, 'Input_State_weight_enc2')
        self.bi_enc2 = self.init_bias(self.hidden_dim2, 'Input_Hidden_bias_enc2')

        # Similar initialization is repeated for forget gates, output gates, cell states, and decoders
        self.init_decoder_weights()

    def init_decoder_weights(self):
        """Initialize weights and biases for the decoder layers."""
        # Layer 1 Decoder Weights
        self.Wq_dec = self.init_weights(3, self.hidden_dim2, 'Input_Time_weight_dec')
        self.Wi_dec = self.init_weights(self.input_dim, self.hidden_dim2, 'Input_Hidden_weight_dec')
        self.Ui_dec = self.init_weights(self.hidden_dim2, self.hidden_dim2, 'Input_State_weight_dec')
        self.bi_dec = self.init_bias(self.hidden_dim2, 'Input_Hidden_bias_dec')

        # Layer 2 Decoder Weights
        self.Wq_dec2 = self.init_weights(3, self.hidden_dim3, 'Input_Time_weight_dec2')
        self.Wi_dec2 = self.init_weights(self.output_dim2, self.hidden_dim3, 'Input_Hidden_weight_dec2')
        self.Ui_dec2 = self.init_weights(self.hidden_dim3, self.hidden_dim3, 'Input_State_weight_dec2')
        self.bi_dec2 = self.init_bias(self.hidden_dim3, 'Input_Hidden_bias_dec2')

        # Additional decoder weights and biases are initialized similarly

    def init_weights(self, input_dim, output_dim, name=None, std=1.0):
        """Initialize weights with truncated normal distribution."""
        return tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=std / math.sqrt(input_dim)), name=name)

    def init_bias(self, output_dim, name=None):
        """Initialize bias with zeros."""
        return tf.Variable(tf.zeros([output_dim]), name=name)

    def TdLSTM_Encoder_Unit(self, prev_hidden_memory, concat_input):
        """TdLSTM Encoder unit for one timestep."""
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)
        batch_size = tf.shape(concat_input)[0]

        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        T = self.map_elapse_time(t, self.hidden_dim)

        C_ST = tf.nn.sigmoid(tf.matmul(prev_cell, self.W_decomp_enc) + self.b_decomp_enc)
        C_ST_dis = tf.multiply(T, C_ST)
        prev_cell = prev_cell - C_ST + C_ST_dis

        i = tf.sigmoid(tf.matmul(x, self.Wi_enc) + tf.matmul(prev_hidden_state, self.Ui_enc) + self.bi_enc)
        f = tf.sigmoid(tf.matmul(x, self.Wf_enc) + tf.matmul(prev_hidden_state, self.Uf_enc) + self.bf_enc)
        o = tf.sigmoid(tf.matmul(x, self.Wog_enc) + tf.matmul(prev_hidden_state, self.Uog_enc) + self.bog_enc)
        C = tf.nn.tanh(tf.matmul(x, self.Wc_enc) + tf.matmul(prev_hidden_state, self.Uc_enc) + self.bc_enc)

        Ct = f * prev_cell + i * C
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    # Similar structure for Decoder Unit, and other layers...

    def get_representation(self):
        """Get the encoded representation and decoder's initial cell state."""
        all_encoder2_states, all_encoder2_cells = self.get_encoder2_states()
        representation = tf.reverse(all_encoder2_states, [0])[0, :, :]
        decoder_ini_cell = tf.reverse(all_encoder2_cells, [0])[0, :, :]
        return representation, decoder_ini_cell
        scan_input = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim3], tf.float32)  # np.zeros((batch_size, self.hidden_dim3), dtype=np.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, scan_input], 2)  # [seq_length x batch_size x input_dim+1]

        packed_hidden_states = tf.scan(self.TdLSTM_Decoder_Unit, concat_input, initializer=ini_state_cell, name='decoder_states')
        all_decoder_states = packed_hidden_states[:, 0, :, :]
        all_decoder_cells = packed_hidden_states[:, 1, :, :]

        return all_decoder_states, all_decoder_cells

    def get_decoder2_states(self, ini_cell):
        encoder_representation, decoder_ini_cell = self.get_representation()
        batch_size = tf.shape(encoder_representation)[0]
        scan_time = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        
        initial_hidden = tf.zeros([batch_size, self.hidden_dim2], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, encoder_representation], 2)  # [seq_length x batch_size x input_dim+1]

        packed_hidden_states = tf.scan(self.TdLSTM_Decoder_Unit2, concat_input, initializer=ini_state_cell, name='decoder2_states')
        all_decoder_states2 = packed_hidden_states[:, 0, :, :]
        all_decoder_cells2 = packed_hidden_states[:, 1, :, :]

        return all_decoder_states2, all_decoder_cells2

    def map_elapse_time(self, t, hidden_dim):
        T = tf.exp(-tf.maximum(0.0, tf.cast(t, tf.float32)))
        return T

    def train_model(self, input_data, time_data, learning_rate=0.001, epochs=10):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        representation, decoder_ini_cell = self.get_representation()
        
        all_decoder_states, _ = self.get_decoder_states()

        loss = tf.reduce_mean(tf.square(input_data - all_decoder_states))  # Mean Squared Error loss
        
        training_op = optimizer.minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(epochs):
                feed_dict = {self.input: input_data, self.time: time_data, self.keep_prob: 0.8}
                _, l = sess.run([training_op, loss], feed_dict=feed_dict)
                
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {l}')
                    
            return sess.run(representation)

    def predict(self, input_data, time_data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {self.input: input_data, self.time: time_data, self.keep_prob: 1.0}
            representation, decoder_ini_cell = sess.run(self.get_representation(), feed_dict=feed_dict)
            return representation
