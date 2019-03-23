import os

import tensorflow as tf
import numpy as np

import inference
import read

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAIN_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "model/"
MODEL_NAME = "test"

def train(train_images, train_labels, test_images, test_labels):
    with tf.device('/CPU:0'):
        x = tf.placeholder(tf.float32, [BATCH_SIZE, inference.IMAGE_SIZE,
                                        inference.IMAGE_SIZE,
                                        inference.NUM_CHANNELS], name='x-input')
        y_o = tf.placeholder(tf.int32, [BATCH_SIZE], name='y-ori-input')
        y_ = tf.cast(tf.one_hot(y_o, 10, 1, 0), dtype=tf.float32)

        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

        train = tf.Variable(True, trainable=False)

        y = inference.inference(x, train, regularizer)

        global_step = tf.Variable(0, trainable=False)

        variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)


        variable_average_op = variable_average.apply(tf.trainable_variables())
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        #loss = tf.add_n(tf.get_collection('losses'))

        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                   global_step=global_step,
                                                   decay_steps=train_images.shape[0] / BATCH_SIZE,
                                                   decay_rate=LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variable_average_op]):
        #with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            totol_samples = train_images.shape[0]

            for i in range(TRAIN_STEPS):
                #start = (i - 1) * BATCH_SIZE % totol_samples
                start = 0
                end = start + BATCH_SIZE
                xs, ys = train_images[start:end], train_labels[start:end]
                xs = np.reshape(xs, [BATCH_SIZE, inference.IMAGE_SIZE,
                                     inference.IMAGE_SIZE,
                                     inference.NUM_CHANNELS])
                _, loss_value, step, y_predict = sess.run([train_op, loss, global_step, y], feed_dict={x: xs, y_o: ys})
                if i % 10 == 0:
                    print("After %d training steps, loss on training batch is %g" % (step, loss_value))
                    #print(type(y_predict))
                    y_predict = y_predict.argmax(1)
                    count = 0
                    for index in range(BATCH_SIZE):
                        if y_predict[index] == ys[index]:
                            count += 1
                    print("rate is %f%%" % (count / BATCH_SIZE))

                if i % 1000 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

if __name__ == '__main__':
    train_images = read.load_train_images()
    train_labels = read.load_train_labels()

    test_images = read.load_test_images()
    test_labels = read.load_test_labels()

    train(train_images, train_labels, test_images, test_labels)