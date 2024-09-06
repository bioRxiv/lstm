import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import h5py
from TdLSTM_error import TdLSTM_error

# Load synthetic data
data, time, assignments, target = [], [], [], []
with h5py.File("sample.mat") as f:
    for column in f['Data']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Data.append(row_data)
    for column in f['Time']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Time.append(row_data)
    for column in f['Assign']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Assignments.append(row_data)

cell_len = len(Data[0])

def generate_batches(data, time, assign, idx):
    """Generate batches for the input data, time, and assignments."""
    batch_data = np.transpose(data[0][idx])
    batch_time = np.transpose(time[0][idx])
    batch_assign = np.transpose(assign[0][idx])
    return batch_data, batch_time, batch_assign

# Hyperparameters
learning_rate = 1e-3
ae_iters = 2000

# Network parameters
input_dim = data[0][0].shape[0]
hidden_dim = 8
hidden_dim2 = 2
hidden_dim3 = 8

# Initialize TLSTM auto-encoder model
lstm_ae = TdLSTM_AE(input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim, hidden_dim2, hidden_dim3)

# Define loss function and optimizer
loss_ae = lstm_ae.get_reconstruction_loss()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_ae)

# Training the model
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    loss_values = np.zeros(ae_iters)
    
    # Training loop
    for i in range(ae_iters):
        total_loss = 0
        for j in range(cell_len):
            x_batch, t_batch, _ = generate_batches(data, time, assignments, j)
            _, batch_loss = sess.run([optimizer, loss_ae], feed_dict={lstm_ae.input: x_batch, lstm_ae.time: t_batch})
            total_loss += batch_loss
        
        # Average loss for this iteration
        loss_values[i] = total_loss / cell_len
        print(f"Iteration {i+1}/{ae_iters}, Loss: {loss_values[i]:.6f}")
    
    # Extract latent representations
    assign_truth, data_reps = [], []
    
    for c in range(cell_len):
        x_data = np.transpose(data[0][c])
        t_data = np.transpose(time[0][c])
        assign_data = np.transpose(assignments[0][c])
        
        reps, _ = sess.run(lstm_ae.get_representation(), feed_dict={lstm_ae.input: x_data, lstm_ae.time: t_data})
        
        if c == 0:
            data_reps = reps
            assign_truth = assign_data
        else:
            data_reps = np.concatenate((data_reps, reps))
            assign_truth = np.concatenate((assign_truth, assign_data))

# Clustering using KMeans
kmeans = KMeans(n_clusters=4, random_state=0, init='k-means++').fit(data_reps)
centroid_values = kmeans.cluster_centers_

# Visualization of clustering results
plt.figure(1)
plt.scatter(data_reps[:, 0], data_reps[:, 1], c=assign_truth, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=35)
plt.title('TdLSTM Clustering')
plt.show()

