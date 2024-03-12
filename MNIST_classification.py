import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist as mnist
import matplotlib.pyplot as plt

dataset_path = 'MNIST'
input_data = mnist.input_data.read_data_sets(dataset_path, one_hot=True)
train_data = input_data.train
test_data = input_data.test
validation_data = input_data.validation

def conv_net(input_data, weights, biases, dropout):
    input_data = tf.reshape(tensor=input_data, shape=[-1,28,28,1])
    conv1 = tf.nn.conv2d(input=input_data, filter=weights['w1'], strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, biases['b1'])
    relu1 = tf.nn.relu(conv1)
    maxpool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    conv2 = tf.nn.conv2d(input=maxpool1, filter=weights['w2'], strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, biases['b2'])
    relu2 = tf.nn.relu(conv2)
    maxpool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    fc1 = tf.reshape(tensor=maxpool2, shape=[-1,weights['w3'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w3']), biases['b3'])
    fc1 = tf.nn.relu(fc1)
    
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['w4']), biases['b4'])
    
    return out

learning_rate = 0.01
epochs = 5
batch_size = 256
num_batches = np.int32(input_data.train.num_examples/batch_size)
im_dim = 28
n_classes = 10
dropout = 0.8
filter_dim = 5

x = tf.placeholder(dtype=tf.float32, shape=[None, im_dim*im_dim])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(dtype=tf.float32)

weights = {
        'w1': tf.Variable(initial_value=tf.random_normal(shape=[filter_dim, filter_dim, 1, 64])),
        'w2': tf.Variable(initial_value=tf.random_normal(shape=[filter_dim, filter_dim, 64, 128])),
        'w3': tf.Variable(initial_value=tf.random_normal(shape=[np.int32(im_dim*im_dim/16*128), 1024])),
        'w4': tf.Variable(initial_value=tf.random_normal(shape=[1024, n_classes]))
        }

biases = {
        'b1': tf.Variable(initial_value=tf.random_normal(shape=[64])),
        'b2': tf.Variable(initial_value=tf.random_normal(shape=[128])),
        'b3': tf.Variable(initial_value=tf.random_normal(shape=[1024])),
        'b4': tf.Variable(initial_value=tf.random_normal(shape=[n_classes]))
        }

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
correct_pred = tf.equal(tf.argmax(input=pred, axis=1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for j in range(num_batches):
            batch_x, batch_y = train_data.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_x, y:batch_y, keep_prob:1})
            loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1})
            print('Epoch:', i+1, ' iter:', j+1, ' loss:', loss, ' acc:', acc)
            
        print('=====================================================')
        print('Epoch : ', i+1, ' loss : ', loss)
    print('Training Completed')
    
    y1 = sess.run(pred, feed_dict={x:test_data.images[:256], keep_prob:1})
    test_classes = np.argmax(y1, 1)
    test_acc = sess.run(accuracy, feed_dict={x:validation_data.images[:256], y:validation_data.labels[:256], keep_prob:1})
    print('Test Accuracy : ', test_acc)
    
    f, a = plt.subplots(1, 10, figsize=(10,2))
    for i in range(10):
        a[i].imshow(np.reshape(validation_data.images[i], (28,28)))
        print(test_classes[i])
    f.show()
    sess.close()