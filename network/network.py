import tensorflow as tf
import numpy as np
from network.loss_functions import huber_loss


class AlexNetDuel(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        weights_path = 'models/imagenet.npy'
        weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        # print('Loading imagenet weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, weights["conv1"][0], np.random.normal(0, 1) * weights["conv1"][1], k=11, out=96,
                               s=4, p="VALID", trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, weights["conv2"][0], weights["conv2"][1], k=5, out=256, s=1, p="SAME",
                               trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, weights["conv3"][0], weights["conv3"][1], k=3, out=384, s=1, p="SAME",
                               trainable=train_conv)
        self.conv4 = self.conv(self.conv3, weights["conv4"][0], weights["conv4"][1], k=3, out=384, s=1, p="SAME",
                               trainable=train_conv)
        self.conv5 = self.conv(self.conv4, weights["conv5"][0], weights["conv5"][1], k=3, out=256, s=1, p="SAME",
                               trainable=train_conv)
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat = tf.contrib.layers.flatten(self.maxpool5)

        # Advantage Network
        self.fc6_a = self.FullyConnected(self.flat, units_in=1024, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_a = self.FullyConnected(self.fc6_a, units_in=2048, units_out=1024, act='relu', trainable=train_fc7)
        self.fc8_a = self.FullyConnected(self.fc7_a, units_in=1024, units_out=1024, act='relu', trainable=train_fc8)
        self.fc9_a = self.FullyConnected(self.fc8_a, units_in=1024, units_out=512, act='relu', trainable=train_fc9)
        self.fc10_a = self.FullyConnected(self.fc9_a, units_in=512, units_out=num_actions, act='linear', trainable=True)

        # Value Network
        self.fc6_v = self.FullyConnected(self.flat, units_in=1024, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_v = self.FullyConnected(self.fc6_v, units_in=2048, units_out=1024, act='relu', trainable=train_fc7)
        self.fc8_v = self.FullyConnected(self.fc7_v, units_in=1024, units_out=1024, act='relu', trainable=train_fc8)
        self.fc9_v = self.FullyConnected(self.fc8_v, units_in=1024, units_out=512, act='relu', trainable=train_fc9)
        self.fc10_v = self.FullyConnected(self.fc9_v, units_in=512, units_out=1, act='linear', trainable=True)

        self.output = self.fc10_v + tf.subtract(self.fc10_a, tf.reduce_mean(self.fc10_a, axis=1, keep_dims=True))

    def conv(self, input, W, b, k, out, s, p, trainable=True):
        assert (W.shape[0] == k)
        assert (W.shape[1] == k)
        assert (W.shape[3] == out)

        conv_kernel_1 = tf.nn.conv2d(input, tf.Variable(W, trainable), [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W, b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        else:
            assert (1 == 0)


class C3F2(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        # weights_path = 'models/imagenet.npy'
        # weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        # print('Loading imagenet weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, k=7, out=96, s=4, p="VALID", trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, k=5, out=64, s=1, p="VALID", trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, k=3, out=64, s=1, p="SAME", trainable=train_conv)

        self.flat = tf.contrib.layers.flatten(self.conv3)

        # Advantage Network
        self.fc1 = self.FullyConnected(self.flat, units_in=1024, units_out=1024, act='relu', trainable=train_fc6)
        self.fc2 = self.FullyConnected(self.fc1, units_in=1024, units_out=num_actions, act='linear',
                                       trainable=train_fc7)

        self.output = self.fc2

    def conv(self, input, k, out, s, p, trainable=True):

        W = tf.Variable(tf.truncated_normal(shape=(k, k, int(input.shape[3]), out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[out], stddev=0.05), trainable=trainable)

        conv_kernel_1 = tf.nn.conv2d(input, W, [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W, b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        else:
            assert (1 == 0)


class C3F2_REINFORCE_with_baseline(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, k=7, out=96, s=4, p="VALID", trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, k=5, out=64, s=1, p="VALID", trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, k=3, out=64, s=1, p="SAME", trainable=train_conv)

        self.flat = tf.contrib.layers.flatten(self.conv3)

        # Main Network
        self.fc1 = self.FullyConnected(self.flat, units_in=1024, units_out=1024, act='relu', trainable=train_fc6)
        self.fc2 = self.FullyConnected(self.fc1, units_in=1024, units_out=num_actions, act='softmax',
                                       trainable=train_fc7)

        # Baseline Network
        self.fc1_baseline = self.FullyConnected(self.flat, units_in=1024, units_out=1024, act='relu',
                                                trainable=train_fc6)
        self.fc2_baseline = self.FullyConnected(self.fc1_baseline, units_in=1024, units_out=1, act='linear',
                                                trainable=train_fc7)

        self.output = self.fc2
        self.baseline = self.fc2_baseline

    def conv(self, input, k, out, s, p, trainable=True):

        W = tf.Variable(tf.truncated_normal(shape=(k, k, int(input.shape[3]), out), stddev=0.05 / 10, seed=1),
                        trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[out], stddev=0.05 / 10, seed=1), trainable=trainable)

        conv_kernel_1 = tf.nn.conv2d(input, W, [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05 / 10, seed=1), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05 / 10, seed=1), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W, b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        elif act == 'softmax':
            return tf.nn.softmax(tf.nn.xw_plus_b(input, W, b))
        else:
            assert (1 == 0)


class AlexNetConditional(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        weights_path = 'models/imagenet.npy'
        weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        # print('Loading imagenet weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, weights["conv1"][0], weights["conv1"][1], k=11, out=96, s=4, p="VALID",
                               trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, weights["conv2"][0], weights["conv2"][1], k=5, out=256, s=1, p="SAME",
                               trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, weights["conv3"][0], weights["conv3"][1], k=3, out=384, s=1, p="SAME",
                               trainable=train_conv)

        # Divide the network stream from this point onwards

        # One - Main Network
        self.conv4_main = self.conv(self.conv3, weights["conv4"][0], weights["conv4"][1], k=3, out=384, s=1, p="SAME",
                                    trainable=train_conv)
        self.conv5_main = self.conv(self.conv4_main, weights["conv5"][0], weights["conv5"][1], k=3, out=256, s=1,
                                    p="SAME", trainable=train_conv)
        self.maxpool5_main = tf.nn.max_pool(self.conv5_main, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat_main = tf.contrib.layers.flatten(self.maxpool5_main)

        # Advantage Network
        self.fc6_a_main = self.FullyConnected(self.flat_main, units_in=9216, units_out=4096, act='relu',
                                              trainable=train_fc6)
        self.fc7_a_main = self.FullyConnected(self.fc6_a_main, units_in=4096, units_out=2048, act='relu',
                                              trainable=train_fc7)
        self.fc8_a_main = self.FullyConnected(self.fc7_a_main, units_in=2048, units_out=num_actions, act='linear',
                                              trainable=train_fc8)

        # Value Network
        self.fc6_v_main = self.FullyConnected(self.flat_main, units_in=9216, units_out=4096, act='relu',
                                              trainable=train_fc6)
        self.fc7_v_main = self.FullyConnected(self.fc6_v_main, units_in=4096, units_out=2048, act='relu',
                                              trainable=train_fc7)
        self.fc8_v_main = self.FullyConnected(self.fc7_v_main, units_in=2048, units_out=1, act='linear', trainable=True)

        self.output_main = self.fc8_v_main + tf.subtract(self.fc8_a_main,
                                                         tf.reduce_mean(self.fc8_a_main, axis=1, keep_dims=True))

        # Two - Conditional Network
        conv4_cdl_k = np.random.rand(3, 3, 384, 256).astype(np.float32)
        conv4_cdl_b = np.random.rand(256).astype(np.float32)
        self.conv4_cdl = self.conv(self.conv3, conv4_cdl_k, conv4_cdl_b, k=3, out=256, s=1, p="SAME",
                                   trainable=train_conv)
        self.maxpool4_cdl = tf.nn.max_pool(self.conv4_cdl, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat_cdl = tf.contrib.layers.flatten(self.maxpool4_cdl)

        # Advantage Network
        self.fc6_a_cdl = self.FullyConnected(self.flat_cdl, units_in=9216, units_out=2048, act='relu',
                                             trainable=train_fc6)
        self.fc7_a_cdl = self.FullyConnected(self.fc6_a_cdl, units_in=2048, units_out=num_actions, act='linear',
                                             trainable=train_fc7)

        # Value Network
        self.fc6_v_cdl = self.FullyConnected(self.flat_cdl, units_in=9216, units_out=2048, act='relu',
                                             trainable=train_fc6)
        self.fc7_v_cdl = self.FullyConnected(self.fc6_v_cdl, units_in=2048, units_out=1, act='linear',
                                             trainable=train_fc7)

        self.output_cdl = self.fc7_v_cdl + tf.subtract(self.fc7_a_cdl,
                                                       tf.reduce_mean(self.fc7_a_cdl, axis=1, keep_dims=True))

    def conv(self, input, W, b, k, out, s, p, trainable=True):
        assert (W.shape[0] == k)
        assert (W.shape[1] == k)
        assert (W.shape[3] == out)

        conv_kernel_1 = tf.nn.conv2d(input, tf.Variable(W, trainable), [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W, b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        else:
            assert (1 == 0)


class AlexNetDuelPrune(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        # weights_path = 'models/imagenet.npy'
        weights_path = 'models/prune_weights.npy'
        weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        print('Loading pruned weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, weights["conv1W"], weights["conv1b"], k=11, out=64, s=4, p="VALID",
                               trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, weights["conv2W"], weights["conv2b"], k=5, out=192, s=1, p="SAME",
                               trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, weights["conv3W"], weights["conv3b"], k=3, out=288, s=1, p="SAME",
                               trainable=train_conv)
        self.conv4 = self.conv(self.conv3, weights["conv4W"], weights["conv4b"], k=3, out=288, s=1, p="SAME",
                               trainable=train_conv)
        self.conv5 = self.conv(self.conv4, weights["conv5W"], weights["conv5b"], k=3, out=256, s=1, p="SAME",
                               trainable=train_conv)
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat = tf.contrib.layers.flatten(self.maxpool5)

        # Advantage Network
        self.fc6_a = self.FullyConnected(self.flat, units_in=9216, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_a = self.FullyConnected(self.fc6_a, units_in=2048, units_out=1024, act='relu', trainable=train_fc7)
        self.fc8_a = self.FullyConnected(self.fc7_a, units_in=1024, units_out=1024, act='relu', trainable=train_fc8)
        self.fc9_a = self.FullyConnected(self.fc8_a, units_in=1024, units_out=512, act='relu', trainable=train_fc9)
        self.fc10_a = self.FullyConnected(self.fc9_a, units_in=512, units_out=num_actions, act='linear', trainable=True)

        # Value Network
        self.fc6_v = self.FullyConnected(self.flat, units_in=9216, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_v = self.FullyConnected(self.fc6_v, units_in=2048, units_out=1024, act='relu', trainable=train_fc7)
        self.fc8_v = self.FullyConnected(self.fc7_v, units_in=1024, units_out=1024, act='relu', trainable=train_fc8)
        self.fc9_v = self.FullyConnected(self.fc8_v, units_in=1024, units_out=512, act='relu', trainable=train_fc9)
        self.fc10_v = self.FullyConnected(self.fc9_v, units_in=512, units_out=1, act='linear', trainable=True)

        self.output = self.fc10_v + tf.subtract(self.fc10_a, tf.reduce_mean(self.fc10_a, axis=1, keep_dims=True))

    def conv(self, input, W, b, k, out, s, p, trainable=True):
        assert (W.shape[0] == k)
        assert (W.shape[1] == k)
        assert (W.shape[3] == out)

        conv_kernel_1 = tf.nn.conv2d(input, tf.Variable(W, trainable), [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W, b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        else:
            assert (1 == 0)

class AlexNet(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x

        print("-----------------ALEX NET Network Training-------------")

        weights_path = 'models/imagenet.npy'
        weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        print('Loading imagenet weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, weights["conv1"][0], weights["conv1"][1], k=11, out=96, s=4, p="VALID",
                               trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, weights["conv2"][0], weights["conv2"][1], k=5, out=256, s=1, p="SAME",
                               trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, weights["conv3"][0], weights["conv3"][1], k=3, out=384, s=1, p="SAME",
                               trainable=train_conv)
        self.conv4 = self.conv(self.conv3, weights["conv4"][0], weights["conv4"][1], k=3, out=384, s=1, p="SAME",
                               trainable=train_conv)
        self.conv5 = self.conv(self.conv4, weights["conv5"][0], weights["conv5"][1], k=3, out=256, s=1, p="SAME",
                               trainable=train_conv)
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat = tf.contrib.layers.flatten(self.maxpool5)

        self.fc6 = self.FullyConnected(self.flat, units_in=9216, units_out=4096, act='relu', trainable=train_fc6)
        self.fc7 = self.FullyConnected(self.fc6, units_in=4096, units_out=2048, act='relu', trainable=train_fc7)
        self.fc8 = self.FullyConnected(self.fc7, units_in=2048, units_out=2048, act='relu', trainable=train_fc8)
        self.fc9 = self.FullyConnected(self.fc8, units_in=2048, units_out=1024, act='relu', trainable=train_fc9)
        self.fc10 = self.FullyConnected(self.fc9, units_in=1024, units_out=num_actions, act='linear', trainable=True)

        self.output = self.fc10
        print(self.conv1)
        print(self.conv2)
        print(self.conv3)
        print(self.conv4)
        print(self.conv5)
        print(self.fc6)
        print(self.fc7)
        print(self.fc8)
        print(self.fc9)
        print(self.fc10)

    def conv(self, input, W, b, k, out, s, p, trainable=True):
        assert (W.shape[0] == k)
        assert (W.shape[1] == k)
        assert (W.shape[3] == out)

        conv_kernel_1 = tf.nn.conv2d(input, tf.Variable(W, trainable), [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.truncated_normal(shape=(units_in, units_out), stddev=0.05)
        b = tf.truncated_normal(shape=[units_out], stddev=0.05)

        if act == 'relu':
            return tf.nn.relu_layer(input, tf.Variable(W, trainable), tf.Variable(b, trainable))
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, tf.Variable(W, trainable), tf.Variable(b, trainable))
        else:
            assert (1 == 0)

class C3F2_ActorCriticShared(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        # # weights_path = 'models/imagenet.npy'
        # weights_path = 'models/prune_weights.npy'
        # weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        # print('Loading pruned weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, k=7, out=96, s=4, p="VALID", trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, k=5, out=64, s=1, p="VALID", trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, k=3, out=64, s=1, p="SAME", trainable=train_conv)

        self.flat = tf.contrib.layers.flatten(self.conv3)

        self.fc1_values = self.FullyConnected(self.flat, units_in=1024, units_out=1024, act='relu', trainable=train_fc6)
        self.fc2_values = self.FullyConnected(self.fc1_values, units_in=1024, units_out=1, act='linear',
                                       trainable=train_fc7)

        self.fc1_actions = self.FullyConnected(self.flat, units_in=1024, units_out=1024, act='relu', trainable=train_fc6)
        self.fc2_actions = self.FullyConnected(self.fc1_actions, units_in=1024, units_out=num_actions, act='softmax',
                                       trainable=train_fc7)

        self.action_probs = self.fc2_actions
        self.state_value = self.fc2_values

    def conv(self, input, k, out, s, p, trainable=True):

        W = tf.Variable(tf.truncated_normal(shape=(k, k, int(input.shape[3]), out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[out], stddev=0.05), trainable=trainable)

        conv_kernel_1 = tf.nn.conv2d(input, W, [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05 / 10, seed=1), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05 / 10, seed=1), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W, b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        elif act == 'softmax':
            return tf.nn.softmax(tf.nn.xw_plus_b(input, W, b))
        else:
            assert (1 == 0)

class AlexNetNotInitialized(tf.keras.Model):
    def __init__(self, num_actions, train_type='all'):
        super(AlexNetNotInitialized, self).__init__()

        # Define the trainability of layers based on train_type
        train_conv = train_fc6 = train_fc7 = train_fc8 = True
        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        # Convolutional layers with randomly initialized weights
        self.conv1 = tf.keras.layers.Conv2D(
            filters=96, kernel_size=11, strides=4, padding="valid", activation='relu',
            kernel_initializer='he_normal', trainable=train_conv
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="valid")

        self.conv2 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=5, strides=1, padding="same", activation='relu',
            kernel_initializer='he_normal', trainable=train_conv
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="valid")

        self.conv3 = tf.keras.layers.Conv2D(
            filters=384, kernel_size=3, strides=1, padding="same", activation='relu',
            kernel_initializer='he_normal', trainable=train_conv
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=384, kernel_size=3, strides=1, padding="same", activation='relu',
            kernel_initializer='he_normal', trainable=train_conv
        )
        self.conv5 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, strides=1, padding="same", activation='relu',
            kernel_initializer='he_normal', trainable=train_conv
        )
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="valid")

        # Flatten the feature map after the last pooling layer
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers with randomly initialized weights
        self.fc6 = tf.keras.layers.Dense(
            units=4096, activation='relu', kernel_initializer='he_normal', trainable=train_fc6
        )
        self.fc7 = tf.keras.layers.Dense(
            units=2048, activation='relu', kernel_initializer='he_normal', trainable=train_fc7
        )
        self.fc8 = tf.keras.layers.Dense(
            units=1024, activation='relu', kernel_initializer='he_normal', trainable=train_fc8
        )

        # Output layer for action predictions
        self.output_layer = tf.keras.layers.Dense(
            units=num_actions, activation='linear', kernel_initializer='he_normal', trainable=True
        )

    def call(self, inputs, training=False):
        # Convolutional layers
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)

        # Output layer
        output = self.output_layer(x)
        return output

class MobileNet(tf.keras.Model):
    def __init__(self, num_actions, input_shape=(227, 227, 3), width_multiplier=1.0):
        super(MobileNet, self).__init__()
        self.num_actions = num_actions
        self.width_multiplier = width_multiplier
        self.build_model(input_shape)

    def build_model(self, input_shape):
        # Initial convolution layer
        self.conv1 = tf.keras.layers.Conv2D(
            filters=int(32 * self.width_multiplier), kernel_size=3, strides=2, padding='same',
            activation='relu', kernel_initializer='he_normal', input_shape=input_shape
        )

        # Depthwise Separable Convolutions
        self.dw_sep_layers = [
            self.depthwise_separable_conv(
                int(32 * self.width_multiplier), int(64 * self.width_multiplier), strides=1, layer_name="conv_dw2"
            ),
            self.depthwise_separable_conv(
                int(64 * self.width_multiplier), int(128 * self.width_multiplier), strides=2, layer_name="conv_dw3"
            ),
            # Add more layers as needed
        ]

        # Global Average Pooling and output layer
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(
            units=self.num_actions, activation='linear', kernel_initializer='he_normal'
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)

        # Apply depthwise separable convolution layers
        for layer in self.dw_sep_layers:
            x = layer(x)

        x = self.global_pool(x)
        output = self.output_layer(x)
        return output

    def depthwise_separable_conv(self, in_channels, out_channels, strides, layer_name):
        return tf.keras.Sequential([
            # Depthwise convolution
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, strides=strides, padding='same', depth_multiplier=1,
                depthwise_initializer=tf.keras.initializers.HeNormal(), activation='relu'
            ),
            # Pointwise convolution
            tf.keras.layers.Conv2D(
                filters=out_channels, kernel_size=1, strides=1, padding='same',
                kernel_initializer=tf.keras.initializers.HeNormal(), activation='relu'
            ),
        ], name=layer_name)