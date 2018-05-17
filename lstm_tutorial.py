import tensorflow as tf
import numpy as np
import collections
import os
import argparse
import datetime as dt

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "/home/ursin/development/eth_project/lstm-tutorial/data"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    """
    The idea of this method is to get the complete word-corpus and then give every word an ID. Words which often appear
    should get small ID's
    """
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    """
    Replace every word with it's ID (as decided in build_vocab())
    """
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    # before tf.reshape: raw_data = [2234, 32234, 34 ... 34534] <-- 929589 values
    # after tf.reshape: data = [2234, 32234, 34 ... 34534]
    #                          [3453, 45444, 55 ... 88636] <---- 46479 values x 20 batches (rows)

    # epoch_size: how many steps (iterations) to go through all data once? batch_len divided by num_steps (35)
    epoch_size = (batch_len - 1) // num_steps

    # extract the x and y data: always 35 per time and the y data shifted by 1 (see blog for an example)
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y


class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # batch_producer() return x and y
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)


# create the main model
class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # create the word embeddings. We learn an embedding for each word in the vocabulary.
        # The embedding-vector has a depth of hidden_size, which is arbitrary. We train the embeddings together with
        # the overall-training, but in general we could also pre-train them.
        with tf.device("/cpu:0"):
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            # lookup the input data in the embeddings --> inputs is afterwards the matching embedding to the x-data
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        if is_training and dropout < 1:
            # this is a higher-level tensorflow operations (comparable with Keras) which adds a dropout for the
            # whole inputs tensor. https://www.tensorflow.org/api_docs/python/tf/nn/dropout
            inputs = tf.nn.dropout(inputs, dropout)

        # We want to load the final state of the learning on the previous batch when we begin with learning the next
        # batch (the only exception is at the beginning of a new epoch - there we want to reset the state to 0).

        # the size is   num_layers * the previous output from the cell h^t-1 *
        #               previous state variable s^t-1 * batch-size * hidden_size
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        # bring the state (described above) to a proper LSTM-tuple format, one per layer
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled. Important: in fact we create a lot more than just one LSTM-Cell,
        # see https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # output contains the output over one batch * num_steps (20 * 35), where each value has 65 dimensions (hidden_size)
        # self.state is the value we feed into the next batch via initial_state (except when we start a new epoch)

        # we flatten down the output for the softmax to 700 * 650. (the 700 is 35 * 20, so just flatten the batches)
        output = tf.reshape(output, [-1, hidden_size])

        # for the softmax we wanna know what's the probability that the output vector (650 --> hidden_size) is a
        # specific word of the vocabular. That's why we need to have the weight-matrix 'w' and biaas 'b' with
        # the same size as the whole dictionary (10'000 with this test-data)
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))

        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches. Simply a cross entropy loss over all batches
        # and the whole sequence. The third parameter is the weighting, which we don't use.
        # See: https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self.cost = tf.reduce_sum(loss)

        # get the prediction accuracy. The softmax will return probabilities for each word in vocab. NOTE: this is not
        # the loss-function, but used to calculate the accuracy. Even though this two tasks are quite the same,
        # it's important to note that cross entropy is a loss function whereas softmax is an activation function.
        # See: https://www.quora.com/Is-the-softmax-loss-the-same-as-the-cross-entropy-loss/answer/Rafael-Pinto-4?share=4315878f&srid=3ibpF
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))

        # only get the word with the highest probability, which we then use as our predicted word.
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))

        # calculate the accuracy by reducing over all those 1 (correct prediction) and 0 (incorrect predictions) -
        # remember, only in this one batch
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # the next part is kind of an optimization of a simple "out of the box optimizer". It's only done for training.
        if not is_training:
            return

        # we will decrease the learning rate during training, so we need to get it here.
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()

        # clip the gradient during back-propagation to avoid a gradient explosion. Seems to increase performance they say...
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # here we do the gradient descend (kind of manually, with the clipped gradients from before). Not that we use the cost-value calculated before
        # via gradients. So the train_op trains based on the calculated costs.
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        # the next 2 lines makes it possible to change the learning rate during training.
        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        """
        With this line we can set a new learning rate during training. We will do this at the beginning of each epoch.
        """
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def train(train_data, vocabulary_size, num_layers, num_epochs, batch_size, model_save_name,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    # setup data and models
    training_input = Input(batch_size=batch_size, num_steps=35, data=train_data)
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocabulary_size,
              num_layers=num_layers)

    # needs to run after the tensorflow-graph has been created (in the constructor of the Mode())
    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay
    with tf.Session() as sess:
        # start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        # the Coordinator/queue-runner is used in the batch_producer().
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            # after max_lr_epoch (default 10) we start to decrease the learning rate (0.93^2, 0.93^3, 0.93^4, etc...)
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            # m.assign_lr(sess, learning_rate)
            # print(m.learning_rate.eval(), new_lr_decay)

            # as described above, this variable is only used to transfer the state in the RNN over multiple batches,
            # but in the begin of every epoch we reset it.
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
            curr_time = dt.datetime.now()

            # iterate over all the minibatches, which are given by batch-size and num steps (20*35)
            for step in range(training_input.epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])

                # don't print at every minibatch, but only every 50 batches.
                if step % print_iter != 0:
                    # we actually only run the three tf-methods referenced with m.cost, m.train_op and m.state. Remember that we use the current_state to
                    # hand over the state of the cells between mini-batches (see the feed_dict parameter)
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()

                    # same as above, but we also call the m.accuracy function to calculate the accuracy. Only necessary for printing (to see the progress).
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch,
                                                                                                               step,
                                                                                                               cost,
                                                                                                               acc,
                                                                                                               seconds))

            # save a model checkpoint
            saver.save(sess, data_path + '\\' + model_save_name, global_step=epoch)
        # do a final save
        saver.save(sess, data_path + '\\' + model_save_name + '-final')
        # close threads
        coord.request_stop()
        coord.join(threads)


def test(model_path, test_data, reversed_dictionary):
    test_input = Input(batch_size=20, num_steps=35, data=test_data)
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocabulary,
              num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        saver.restore(sess, model_path)
        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        check_batch_idx = 25
        acc_check_thresh = 5
        accuracy = 0
        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true_vals, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy],
                                                               feed_dict={m.init_state: current_state})
                pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
                true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
                print("True values (1st line) vs predicted values (2nd line):")
                print(" ".join(true_vals_string))
                print(" ".join(pred_string))
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc
        print("Average accuracy: {:.3f}".format(accuracy / (num_acc_batches - acc_check_thresh)))
        # close threads
        coord.request_stop()
        coord.join(threads)


if args.data_path:
    data_path = args.data_path
train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()  # vocabulary is only the size
if args.run_opt == 1:
    train(train_data, vocabulary, num_layers=2, num_epochs=100, batch_size=20,
          model_save_name='two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr')
else:
    trained_model = args.data_path + "\\two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr-38"
    test(trained_model, test_data, reversed_dictionary)
