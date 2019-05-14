from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy
import scipy.io
from sklearn.preprocessing import normalize


def _one_hot(input_data):
    eye = 49
    eye = numpy.eye(eye)
    one_hot_output = numpy.sum(eye[input_data-1], axis=1)
    return one_hot_output


def Gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=input_layer.shape, mean=0.0, stddev=std, dtype=tf.float32)
    output = tf.keras.utils.normalize((input_layer+noise), axis=-1, order=1)
    return output


def memory_replay():
    pass


def training_without_onehot():
    pass


def training():
    # Training Parameters

    learning_rate = 0.005
    batch_size = 1

    display_step = 200
    num_epochs = 3

    std = 0.1

    # Network Parameters
    num_input = 7
    timesteps = 30
    num_hidden = 49
    num_classes = 7

    data_set = numpy.load('MarkSix.npy')
    print('Shape of the Dataset:', data_set.shape)
    # data_set = data_set[0:2999+1]

    #data_set = _one_hot(data_set)

    training_set = data_set[0:2500]
    training_set_label = data_set[timesteps:2500 + timesteps]

    testing_set = data_set[2500:3072]

    # testing_set = data_set[2572-timesteps:3072-timesteps]
    # testing_set_label = data_set[2572:3072]
    # testing_set = data_set[2589:2989]
    # testing_set_label = data_set[2599:2999]

    training_set_size = training_set.shape[0]
    testing_set_size = testing_set.shape[0]
    print('Training Set Size:', training_set_size)
    print('Testing Set Size', testing_set_size)

    training_steps = int(training_set_size / batch_size)
    testing_steps = int(testing_set_size / batch_size)

    print('Number of Training Step:', training_steps)
    print('Number of Testing Step:', testing_steps)

    experience_replay_array = numpy.random.random_integers(0, 2500-timesteps, num_epochs*(training_steps-timesteps))
    #experience_replay_array = numpy.arrange(0, num_epochs*(training_steps-timesteps))
    #numpy.shuffle(experience_replay_array)


    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        # Unstack x along axis 1 into 'timesteps' pieces
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input) along axis 1
        # Here it decompose the matrix of 28*28 to 28 timestep of arrays with size 28.
        x = tf.unstack(x, timesteps, 1)

        # Define a lstm cell with tensorflow
        # Forget bias is ranging from 0 to 1. Check manual for the detail
        lstm_cell = rnn.LSTMCell(name='basic_lstm_cell', num_units=num_hidden, forget_bias=1)

        # Get lstm cell output
        # There is another called dynamics rnn.
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        # This will automatically create all the matrices required to map from input to output hidden state
        # The return function is mapping from hidden state to output state, which is a single layer perceptron
        # Therefore, the dimension of the weight is defined as [num_hidden, num_classes]
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def _N_to_N_RNN_cell(x, weights, biases):
        pass

    first_layer_output = tf.nn.relu(RNN(X, weights, biases))
    #dropout_l = tf.nn.dropout(first_layer_output, keep_prob=0.8)
    fc1 = tf.nn.relu(tf.layers.dense(first_layer_output, units=49))
    fc2 = tf.layers.dense(fc1, units=7)
    #prediction = tf.nn.sigmoid(output)
    prediction = tf.round(fc2)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(
        predictions=fc2, labels=Y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    # They are one-hot representation
    correct_pred = tf.equal(tf.round(tf.round(fc2)), Y)#prediction - Y#tf.equal(tf.round(prediction), Y) #tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    epoch = 0
    done_looping = False
    # Start training

    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        #while (epoch < num_epochs) and (not done_looping):
        #    epoch = epoch + 1
        for epoch in range(num_epochs):

            for step in range(training_steps-timesteps):

                temp_step = experience_replay_array[step]
                batch_x = training_set[temp_step:temp_step + timesteps]
                noise = numpy.random.normal(0, std, batch_x.shape)

                batch_x = normalize(batch_x+noise, axis=1)

                batch_y = training_set_label[temp_step:temp_step+timesteps]
                #print(step, 'to' , step+timesteps)
                #print(batch_x.shape)
                #print(batch_y.shape)
                #batch_x = Gaussian_noise_layer(batch_x, 0.01)

                batch_x = batch_x.reshape((batch_size, timesteps, num_input))

                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

                if step % display_step == 0: #or step == 1:
                #if mini_batch display_step == 0 or mini_batch == 1:
                # Calculate batch loss and accuracy
                    iteration = epoch*(training_steps-10) + step
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                    print("Step " + str(iteration) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        #test_len = 128
        Test_accuracy = 0

        for step in range(testing_steps-timesteps+1):
            batch_x = testing_set[step:step + timesteps]

            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            #batch_y = testing_set_label[step:step + timesteps]
            temp_Prediction = sess.run(prediction, feed_dict={X: batch_x})
            if step == 0:
                Prediction = temp_Prediction
            else:
                Prediction = numpy.concatenate((Prediction, temp_Prediction), axis=0)

            Test_accuracy += acc
            #print(temp_Prediction)

        scipy.io.savemat('Prediction.mat', mdict={'Prediction': Prediction})
        Test_accuracy = Test_accuracy/(testing_steps-10)
        print(Test_accuracy)


if __name__ == '__main__':
    training()
