import tensorflow as tf

class TdLSTM:
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, train):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Placeholders for input, labels, time, and dropout probability
        self.input = tf.placeholder(tf.float32, shape=[None, None, self.input_dim], name='input')
        self.labels = tf.placeholder(tf.float32, shape=[None, output_dim], name='labels')
        self.time = tf.placeholder(tf.float32, shape=[None, None], name='time')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Initialize weights and biases
        if train == 1:
            self.init_trainable_weights(output_dim, fc_dim)
        else:
            self.init_non_trainable_weights(output_dim, fc_dim)

    # Weight and bias initialization functions
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.get_variable(name, shape=[input_dim, output_dim], 
                               initializer=tf.random_normal_initializer(0.0, std), regularizer=reg)

    def init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))

    def no_init_weights(self, input_dim, output_dim, name):
        return tf.get_variable(name, shape=[input_dim, output_dim])

    def no_init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim])

    # Initialize trainable weights
    def init_trainable_weights(self, output_dim, fc_dim):
        self.Wi = self.init_weights(self.input_dim, self.hidden_dim, 'Input_Hidden_weight')
        self.Ui = self.init_weights(self.hidden_dim, self.hidden_dim, 'Input_State_weight')
        self.bi = self.init_bias(self.hidden_dim, 'Input_Hidden_bias')

        self.Wf = self.init_weights(self.input_dim, self.hidden_dim, 'Forget_Hidden_weight')
        self.Uf = self.init_weights(self.hidden_dim, self.hidden_dim, 'Forget_State_weight')
        self.bf = self.init_bias(self.hidden_dim, 'Forget_Hidden_bias')

        self.Wog = self.init_weights(self.input_dim, self.hidden_dim, 'Output_Hidden_weight')
        self.Uog = self.init_weights(self.hidden_dim, self.hidden_dim, 'Output_State_weight')
        self.bog = self.init_bias(self.hidden_dim, 'Output_Hidden_bias')

        self.Wc = self.init_weights(self.input_dim, self.hidden_dim, 'Cell_Hidden_weight')
        self.Uc = self.init_weights(self.hidden_dim, self.hidden_dim, 'Cell_State_weight')
        self.bc = self.init_bias(self.hidden_dim, 'Cell_Hidden_bias')

        self.W_decomp = self.init_weights(self.hidden_dim, self.hidden_dim, 'Decomposition_Hidden_weight')
        self.b_decomp = self.init_bias(self.hidden_dim, 'Decomposition_Hidden_bias_enc')

        self.Wo = self.init_weights(self.hidden_dim, fc_dim, 'Fc_Layer_weight')
        self.bo = self.init_bias(fc_dim, 'Fc_Layer_bias')

        self.W_softmax = self.init_weights(fc_dim, output_dim, 'Output_Layer_weight')
        self.b_softmax = self.init_bias(output_dim, 'Output_Layer_bias')

    # Initialize non-trainable weights for testing
    def init_non_trainable_weights(self, output_dim, fc_dim):
        self.Wi = self.no_init_weights(self.input_dim, self.hidden_dim, 'Input_Hidden_weight')
        self.Ui = self.no_init_weights(self.hidden_dim, self.hidden_dim, 'Input_State_weight')
        self.bi = self.no_init_bias(self.hidden_dim, 'Input_Hidden_bias')

        self.Wf = self.no_init_weights(self.input_dim, self.hidden_dim, 'Forget_Hidden_weight')
        self.Uf = self.no_init_weights(self.hidden_dim, self.hidden_dim, 'Forget_State_weight')
        self.bf = self.no_init_bias(self.hidden_dim, 'Forget_Hidden_bias')

        self.Wog = self.no_init_weights(self.input_dim, self.hidden_dim, 'Output_Hidden_weight')
        self.Uog = self.no_init_weights(self.hidden_dim, self.hidden_dim, 'Output_State_weight')
        self.bog = self.no_init_bias(self.hidden_dim, 'Output_Hidden_bias')

        self.Wc = self.no_init_weights(self.input_dim, self.hidden_dim, 'Cell_Hidden_weight')
        self.Uc = self.no_init_weights(self.hidden_dim, self.hidden_dim, 'Cell_State_weight')
        self.bc = self.no_init_bias(self.hidden_dim, 'Cell_Hidden_bias')

        self.W_decomp = self.no_init_weights(self.hidden_dim, self.hidden_dim, 'Decomposition_Hidden_weight')
        self.b_decomp = self.no_init_bias(self.hidden_dim, 'Decomposition_Hidden_bias_enc')

        self.Wo = self.no_init_weights(self.hidden_dim, fc_dim, 'Fc_Layer_weight')
        self.bo = self.no_init_bias(fc_dim, 'Fc_Layer_bias')

        self.W_softmax = self.no_init_weights(fc_dim, output_dim, 'Output_Layer_weight')
        self.b_softmax = self.no_init_bias(output_dim, 'Output_Layer_bias')

    # TdLSTM unit
    def TdLSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)
        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Map elapse time and decompose the previous cell state
        T = self.map_elapse_time(t)
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Gates
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)

        # Candidate memory cell and updated memory cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)
        Ct = f * prev_cell + i * C

        # Current hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    # Compute hidden states for all time steps
    def get_states(self):
        batch_size = tf.shape(self.input)[0]
        scan_input = tf.transpose(self.input, perm=[1, 0, 2])
        scan_time = tf.transpose(self.time)
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        initial_state = tf.stack([initial_hidden, initial_hidden])

        # Concatenate time and input
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, scan_input], 2)

        # Get packed hidden states using scan
        packed_hidden_states = tf.scan(self.TLSTM_Unit, concat_input, initializer=initial_state)
        all_states = packed_hidden_states[:, 0, :, :]
        return all_states

    # Compute the output for a single state
    def get_output(self, state):
        output = tf.nn.relu(tf.matmul(state, self.Wo) + self.bo)
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.matmul(output, self.W_softmax) + self.b_softmax
        return output

    # Compute outputs for all time steps
    def get_outputs(self):
        all_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_states)
        output = tf.reverse(all_outputs, [0])[0, :, :]
        return output

    # Compute cost and accuracy
    def get_cost_acc(self):
        logits = self.get_outputs()
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
        y_pred = tf.argmax(logits, axis=1)
        y_true = tf.argmax(self.labels, axis=1)
        return cross_entropy, y_pred, y_true, logits, self.labels

    # Time mapping function to handle time irregularity
    def map_elapse_time(self, t):
        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)
        T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')
        Ones = tf.ones([1, self.hidden_dim], dtype=tf.float32)
        return tf.matmul(T, Ones)
