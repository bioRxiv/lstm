import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from TdLSTM import TdLSTM

tf.compat.v1.disable_eager_execution()

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def convert_one_hot(label_list):
    for i in range(len(label_list)):
        label_list[i] = np.expand_dims(label_list[i], axis=2)
        sec_col = 1 - label_list[i]
        label_list[i] = np.concatenate([label_list[i], sec_col], axis=2)
    return label_list

def train_model(sess, lstm, optimizer, data_train_batches, labels_train_batches, elapsed_train_batches, train_dropout_prob, epochs):
    number_train_batches = len(data_train_batches)
    
    for epoch in range(epochs):
        total_cost = 0
        for i in range(number_train_batches):
            batch_xs = data_train_batches[i]
            batch_ys = labels_train_batches[i]
            batch_ts = np.reshape(elapsed_train_batches[i], [elapsed_train_batches[i].shape[0], elapsed_train_batches[i].shape[2]])

            sess.run(optimizer, feed_dict={
                lstm.input: batch_xs, 
                lstm.labels: batch_ys,
                lstm.keep_prob: train_dropout_prob, 
                lstm.time: batch_ts
            })

    print("Training is complete!")

def evaluate_model(sess, lstm, data_batches, labels_batches, elapsed_batches, dropout_prob):
    number_batches = len(data_batches)
    Y_pred, Y_true, Labels, Logits = [], [], [], []
    
    for i in range(number_batches):
        batch_xs = data_batches[i]
        batch_ys = labels_batches[i]
        batch_ts = np.reshape(elapsed_batches[i], [elapsed_batches[i].shape[0], elapsed_batches[i].shape[2]])

        y_pred_batch, y_true_batch, logits_batch, labels_batch = sess.run(lstm.get_cost_acc(), feed_dict={
            lstm.input: batch_xs,
            lstm.labels: batch_ys,
            lstm.keep_prob: dropout_prob,
            lstm.time: batch_ts
        })

        Y_true.append(y_true_batch)
        Y_pred.append(y_pred_batch)
        Labels.append(labels_batch)
        Logits.append(logits_batch)

    Y_true = np.concatenate(Y_true)
    Y_pred = np.concatenate(Y_pred)
    Labels = np.concatenate(Labels)
    Logits = np.concatenate(Logits)

    acc = accuracy_score(Y_true, Y_pred)
    auc_micro = roc_auc_score(Labels, Logits, average='micro')
    auc_macro = roc_auc_score(Labels, Logits, average='macro')

    return acc, auc_micro, auc_macro

def training(path, learning_rate, training_epochs, train_dropout_prob, hidden_dim, fc_dim, key, model_path):
    # Load data
    data_train_batches = load_pkl(f'{path}/data_train.pkl')
    elapsed_train_batches = load_pkl(f'{path}/elapsed_train.pkl')
    labels_train_batches = load_pkl(f'{path}/label_train.pkl')

    input_dim = data_train_batches[0].shape[2]
    output_dim = labels_train_batches[0].shape[1]

    lstm = TdLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    cross_entropy, _, _, _, _ = lstm.get_cost_acc()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        train_model(sess, lstm, optimizer, data_train_batches, labels_train_batches, elapsed_train_batches, train_dropout_prob, training_epochs)
        saver.save(sess, model_path)

        acc, auc_micro, auc_macro = evaluate_model(sess, lstm, data_train_batches, labels_train_batches, elapsed_train_batches, train_dropout_prob)
        print(f"Train Accuracy = {acc:.3f}")
        print(f"Train AUC Micro = {auc_micro:.3f}")
        print(f"Train AUC Macro = {auc_macro:.3f}")

def testing(path, hidden_dim, fc_dim, key, model_path):
    # Load data
    data_test_batches = load_pkl(f'{path}/data_test.pkl')
    elapsed_test_batches = load_pkl(f'{path}/elapsed_test.pkl')
    labels_test_batches = load_pkl(f'{path}/label_test.pkl')

    input_dim = data_test_batches[0].shape[2]
    output_dim = labels_test_batches[0].shape[1]

    lstm = TdLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_path)

        acc, auc_micro, auc_macro = evaluate_model(sess, lstm, data_test_batches, labels_test_batches, elapsed_test_batches, 1.0)
        print(f"Test Accuracy = {acc:.3f}")
        print(f"Test AUC Micro = {auc_micro:.3f}")
        print(f"Test AUC Macro = {auc_macro:.3f}")

def main(argv):
    training_mode = int(argv[0])
    path = argv[1]

    if training_mode == 1:
        learning_rate = float(argv[2])
        training_epochs = int(argv[3])
        dropout_prob = float(argv[4])
        hidden_dim = int(argv[5])
        fc_dim = int(argv[6])
        model_path = argv[7]
        training(path, learning_rate, training_epochs, dropout_prob, hidden_dim, fc_dim, training_mode, model_path)
    else:
        hidden_dim = int(argv[2])
        fc_dim = int(argv[3])
        model_path = argv[4]
        testing(path, hidden_dim, fc_dim, training_mode, model_path)

if __name__ == "__main__":
    main(sys.argv[1:])
