# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechat: whatshowlove
@ software: PyCharm
@ file: neural_style
@ time: 18-4-10
"""

import tensorflow as tf
import reader
import _reader
from preprocessing import preprocessing_factory
from nets import nets_factory
import losses
import time
import os
import utils

# tf.app.flags.DEFINE_float("CONTENT_WEIGHT", 5e0, "Weight for content features loss")
# tf.app.flags.DEFINE_float("STYLE_WEIGHT", 1e2, "Weight for style features loss")
# tf.app.flags.DEFINE_float("TV_WEIGHT", 1e-5, "Weight for total variation loss")
# tf.app.flags.DEFINE_string("VGG_MODEL", "pretrained/vgg_16.ckpt", "vgg model params path")
# tf.app.flags.DEFINE_list("CONTENT_LAYERS", ["vgg_16/conv3/conv3_3"],
#                            "Which VGG layer to extract content loss from")
# tf.app.flags.DEFINE_list("STYLE_LAYERS", ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
#                                           "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"],
#                            "Which layers to extract style from")
# tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
# tf.app.flags.DEFINE_string("STYLE_IMAGE", "img/picasso.jpg", "Styles to train")
# tf.app.flags.DEFINE_float("STYLE_SCALE", 1.0, "Scale styles. Higher extracts smaller features")
# tf.app.flags.DEFINE_float("LEARNING_RATE", 10., "Learning rate")
# tf.app.flags.DEFINE_string("CONTENT_IMAGE", "img/dancing.jpg", "Content image to use")
# tf.app.flags.DEFINE_boolean("RANDOM_INIT", True, "Start from random noise")
# tf.app.flags.DEFINE_integer("NUM_ITERATIONS", 1000, "Number of iterations")
# # reduce image size because of cpu training
# tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")

#######################################################################
tf.app.flags.DEFINE_string("loss_model", 'vgg_16', "loss model name")
tf.app.flags.DEFINE_string("naming", 'test', "model_name")
tf.app.flags.DEFINE_string("loss_model_file", "pretrained/vgg_16.ckpt", "pretrained model")
tf.app.flags.DEFINE_string("checkpoint_exclude_scopes", "vgg_16/fc", "ignore variables")
tf.app.flags.DEFINE_float("content_weight", 5, "Weight for content features loss")
tf.app.flags.DEFINE_float("style_weight", 100, "Weight for style features loss")
tf.app.flags.DEFINE_float("tv_weight", 0.0, "Weight for total variation loss")
tf.app.flags.DEFINE_integer("image_size", 256, "Size of output image")
tf.app.flags.DEFINE_list("content_layers", ["vgg_16/conv3/conv3_3"],
                         "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_list("style_layers", ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
                                          "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"],
                         "Which layers to extract style from")
tf.app.flags.DEFINE_string("model_path", 'models', "path to save model")
tf.app.flags.DEFINE_string("content_image", "img/dancing.jpg", "Content image to use")
tf.app.flags.DEFINE_string("style_image", "img/picasso.jpg", "Styles to train")
tf.app.flags.DEFINE_float("learning_rate", 10, "Learning rate")
tf.app.flags.DEFINE_integer("step", 100, "Number of iterations")


FLAGS = tf.app.flags.FLAGS

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.stack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.stack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# TODO: Okay to flatten all style images into one gram?
def gram(layer):
    shape = tf.shape(layer)
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.stack([-1, num_filters]))
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(size)

    return gram

# TODO: Different style scales per image.
def get_style_features(style_paths, style_layers):
    with tf.Graph().as_default() as g:
        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False)

        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False)
        image = tf.expand_dims(
            reader.get_image(FLAGS.style_image, FLAGS.image_size, FLAGS.image_size, image_preprocessing_fn), 0)
        # image = tf.expand_dims(
        #     _reader.get_image(FLAGS.content_image, FLAGS.image_size), 0)
        _, endpoints = network_fn(image, spatial_squeeze=False)

        features = []
        for layer in style_layers:
            features.append(gram(endpoints[layer]))

        with tf.Session() as sess:
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)
            return sess.run(features)

def get_content_features(content_path, content_layers):
    with tf.Graph().as_default() as g:

        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False)

        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False)
        image = tf.expand_dims(
            reader.get_image(FLAGS.content_image, FLAGS.image_size, FLAGS.image_size, image_preprocessing_fn), 0)
        # image = tf.expand_dims(
        #     _reader.get_image(FLAGS.content_image, FLAGS.image_size), 0)
        _, endpoints = network_fn(image, spatial_squeeze=False)
        layers = []
        for layer in content_layers:
            layers.append(endpoints[layer])

        with tf.Session() as sess:
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)
            return sess.run(layers + [image])


def main(argv=None):
    # style_features_t = losses.get_style_features(FLAGS)

    # Make sure the training path exists.
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)


    """get features"""
    style_features_t = get_style_features(FLAGS.style_image, FLAGS.style_layers)
    res = get_content_features(FLAGS.content_image, FLAGS.content_layers)
    content_features_t, image_t = res[:-1], res[-1]
    image = tf.constant(image_t)
    random = tf.random_normal(image_t.shape)
    initial = tf.Variable(image)

    """Build Network"""
    network_fn = nets_factory.get_network_fn(
        FLAGS.loss_model,
        num_classes=1,
        is_training=True)

    image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
        FLAGS.loss_model,
        is_training=False)
    preprocess_content_image = tf.expand_dims(
        reader.get_image(FLAGS.content_image, FLAGS.image_size, FLAGS.image_size, image_preprocessing_fn), 0)
    # preprocess_content_image = tf.expand_dims(
    #     _reader.get_image(FLAGS.content_image, FLAGS.image_size), 0)
    # preprocess_style_image = tf.expand_dims(
    #     reader.get_image(FLAGS.style_image, FLAGS.image_size, FLAGS.image_size, image_preprocessing_fn), 0)
    _, endpoints_dict = network_fn(preprocess_content_image, spatial_squeeze=False)
    """build loss"""
    content_loss = 0
    for content_features, layer in zip(content_features_t, FLAGS.content_layers):
        layer_size = tf.size(content_features)
        content_loss += tf.nn.l2_loss(endpoints_dict[layer] - content_features) / tf.to_float(layer_size)
    content_loss = FLAGS.content_weight * content_loss / len(FLAGS.content_layers)

    style_loss = 0
    for style_gram, layer in zip(style_features_t, FLAGS.style_layers):
        layer_size = tf.size(style_gram)
        style_loss += tf.nn.l2_loss(gram(endpoints_dict[layer]) - style_gram) / tf.to_float(layer_size)
        # style_loss += (gram(endpoints_dict[layer]) - style_gram)
    style_loss = FLAGS.style_weight * style_loss


    tv_loss = FLAGS.tv_weight * total_variation_loss(initial)
    total_loss = content_loss + style_loss + tv_loss
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)
    output_image = tf.image.encode_png(tf.saturate_cast(tf.squeeze(initial) + reader.mean_pixel, tf.uint8))

    with tf.Session() as sess:
        init_func = utils._get_init_fn(FLAGS)
        init_func(sess)
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for step in range(FLAGS.step):
            _, loss_t, cl, sl = sess.run([train_op, total_loss, content_loss, style_loss])
            elapsed = time.time() - start_time
            start_time = time.time()
            print(step, elapsed, loss_t, cl, sl)
        image_t = sess.run(output_image)
        with open('out.png', 'wb') as f:
            f.write(image_t)

if __name__ == '__main__':
    tf.app.run()



