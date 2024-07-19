import tensorflow as tf
import numpy as np
from absl import flags

import host_call as host_call_ported

flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
         "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
         "url.")

flags.DEFINE_string("model_dir", None, "Estimator model_dir")
flags.DEFINE_string("export_dir", None, "Export model_dir dir")
flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
flags.DEFINE_integer("iterations", 100, "Number of iterations per TPU training loop.")
flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
flags.DEFINE_integer("test_steps", 5, "Total number of training steps.")
flags.DEFINE_string("train_file", None, "Train file")
flags.DEFINE_string("test_file", None, "Test file")
flags.DEFINE_integer("batch_size", 1024, "Batch size")
flags.DEFINE_bool("tpu_enable_host_call", False, "Enable TPUEstimator host_call.")
flags.DEFINE_integer("level_number", 30, "Number of iterations.")
flags.DEFINE_bool("add_summaries", False, "Add TF summaries.")
flags.DEFINE_integer("variable_number", 8, "Variable number.")
flags.DEFINE_integer("clause_number", 80, "Clause number (maximal, to determine tensor shape).")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate.")
flags.DEFINE_bool("train_files_gzipped", False, "Are train files gzipped.")
flags.DEFINE_bool("test_files_gzipped", False, "Are test files gzipped.")
flags.DEFINE_bool("export_model", False, "Export saved model for prediction.")
flags.DEFINE_bool("attention", True, "Should attention be used.")

FLAGS = flags.FLAGS

DEFAULT_SETTINGS = {

    # Only for not SR
    "MIN_CLAUSE_NUM": 1,

    "SR_GENERATOR": True,

    # Neural net
    "EMBEDDING_SIZE": 128,

    "POS_NEG_ACTIVATION": None,
    "HIDDEN_LAYERS": [128, 128],
    "HIDDEN_ACTIVATION": tf.nn.relu,
    "EMBED_ACTIVATION": tf.nn.tanh,

    "POLICY_LOSS_WEIGHT": 1,
    "SAT_LOSS_WEIGHT": 1,

    "NEPTUNE_ENABLED": False,
    "BOARD_WRITE_GRAPH": False,

    # Size of dataset

    "SAMPLES": 10 ** 4,

    # Multiprocessing
    # "PROCESSOR_NUM": None,  # defaults to all processors,

    "MODEL_DIR": "gs://ng-training-data",  # this should go to train_policy
}

class Graph:
    def __init__(self, settings, features=None, labels=None):
        if features is None:
            self.inputs = tf.keras.Input(shape=(None, None, 2), name='inputs')
        else:
            self.inputs = features

        def assert_shape(a, b):
            tf.debugging.assert_equal(tf.shape(a), tf.shape(b))

        if labels is None:
            self.policy_labels = tf.keras.Input(shape=(None, 2), name='policy_labels')
            self.sat_labels = tf.keras.Input(shape=(), name='sat_labels')
        else:
            self.sat_labels = labels["sat"]
            self.policy_labels = labels["policy"]

        batch_size = tf.shape(self.inputs)[0]
        variable_num = tf.shape(self.inputs)[1]
        clause_num = tf.shape(self.inputs)[2]
        assert_shape(variable_num, [])
        assert_shape(clause_num, [])

        variables_per_clause = tf.reduce_sum(self.inputs, axis=[1, 3])
        clauses_per_variable = tf.reduce_sum(self.inputs, axis=[2, 3])
        assert_shape(variables_per_clause, [batch_size, None])
        assert_shape(clauses_per_variable, [batch_size, None])

        positive_connections, negative_connections = tf.split(self.inputs, 2, axis=3)
        positive_connections = tf.squeeze(positive_connections, axis=3)
        negative_connections = tf.squeeze(negative_connections, axis=3)
        all_connections = tf.concat([positive_connections, negative_connections], axis=1)
        assert_shape(positive_connections, [batch_size, None, None])
        assert_shape(negative_connections, [batch_size, None, None])

        self.loss = 0.0

        EMBEDDING_SIZE = settings["EMBEDDING_SIZE"]
        HIDDEN_LAYERS = settings["HIDDEN_LAYERS"]
        HIDDEN_ACTIVATION = settings["HIDDEN_ACTIVATION"]
        SAT_LOSS_WEIGHT = settings["SAT_LOSS_WEIGHT"]
        POLICY_LOSS_WEIGHT = settings["POLICY_LOSS_WEIGHT"]
        LEVEL_NUMBER = FLAGS.level_number
        EMBED_ACTIVATION = settings["EMBED_ACTIVATION"]
        ATTENTION = FLAGS.attention

        def basic_MLP(source, name, target_dimension=None, hidden_layers=None, end_activation=None):
            if hidden_layers is None:
                hidden_layers = HIDDEN_LAYERS
            if target_dimension is None:
                target_dimension = EMBEDDING_SIZE
            last_hidden = source
            for index, size in enumerate(hidden_layers):
                last_hidden = tf.keras.layers.Dense(size, activation=HIDDEN_ACTIVATION, name='{}_hidden_{}'.format(name, index))(last_hidden)
            return tf.keras.layers.Dense(target_dimension, activation=end_activation, name='{}'.format(name))(last_hidden)

        def aggregate(Q, K, V, conn):
            if ATTENTION:
                QtimesK = tf.matmul(Q, K, transpose_b=True)
                norm_weights = tf.multiply(tf.sigmoid(QtimesK), conn)
            else:
                unnorm_weights = conn
                norm_weights = unnorm_weights / (1.0 + tf.reduce_sum(unnorm_weights, axis=-1, keepdims=True))
            aggr_V = tf.matmul(norm_weights, V)
            return aggr_V

        initial_var_embedding = tf.Variable(
            initial_value=tf.random.uniform(shape=[EMBEDDING_SIZE], minval=-1., maxval=1.),
            name='init_var_embedding')
        positive_literal_embeddings = tf.tile(
            tf.reshape(initial_var_embedding, [1, 1, EMBEDDING_SIZE]),
            [batch_size, variable_num, 1])
        negative_literal_embeddings = tf.tile(
            tf.reshape(initial_var_embedding, [1, 1, EMBEDDING_SIZE]),
            [batch_size, variable_num, 1])

        initial_clause_embedding = tf.Variable(
            initial_value=tf.random.uniform(shape=[1, 1, EMBEDDING_SIZE], minval=-1., maxval=1.),
            name='init_clause_embedding')
        clause_embeddings = tf.tile(
            tf.reshape(initial_clause_embedding, [1, 1, EMBEDDING_SIZE]),
            [batch_size, clause_num, 1])

        self.sat_list = []  # sat prediction level by level (rounded)
        self.policy_list = []  # policy prediction level by level (rounded)

        for level in range(LEVEL_NUMBER + 1):
            if level >= 1:
                assert_shape(positive_literal_embeddings, [batch_size, None, EMBEDDING_SIZE])
                assert_shape(negative_literal_embeddings, [batch_size, None, EMBEDDING_SIZE])

                # clause preembeddings
                cls4cls_V = basic_MLP(clause_embeddings, 'cls4cls_V')

                if ATTENTION:
                    cls4lit_Q = basic_MLP(clause_embeddings, 'cls4lit_Q')
                else:
                    cls4lit_Q = None

                all_literal_embeddings = tf.concat([positive_literal_embeddings, negative_literal_embeddings], axis=1)
                if ATTENTION:
                    lit4cls_K = basic_MLP(all_literal_embeddings, 'lit4cls_K')
                else:
                    lit4cls_K = None
                lit4cls_V = basic_MLP(all_literal_embeddings, 'lit4cls_V')

                lit4cls_aggr_V = aggregate(cls4lit_Q, lit4cls_K, lit4cls_V, tf.transpose(all_connections, perm=[0, 2, 1]))

                clause_preembeddings = tf.concat([cls4cls_V, lit4cls_aggr_V], axis=-1)

                # literal preembeddings
                pos4pos_V = basic_MLP(positive_literal_embeddings, 'lit4lit_V')
                neg4neg_V = basic_MLP(negative_literal_embeddings, 'lit4lit_V')

                pos4neg_V = basic_MLP(positive_literal_embeddings, 'neg4neg_V')
                neg4pos_V = basic_MLP(negative_literal_embeddings, 'neg4neg_V')

                if ATTENTION:
                    pos4cls_Q = basic_MLP(positive_literal_embeddings, 'lit4cls_Q')
                    neg4cls_Q = basic_MLP(negative_literal_embeddings, 'lit4cls_Q')

                    cls4lit_K = basic_MLP(clause_embeddings, 'cls4lit_K')
                else:
                    pos4cls_Q, neg4cls_Q, cls4lit_K = None, None, None
                cls4lit_V = basic_MLP(clause_embeddings, 'cls4lit_V')

                cls4pos_aggr_V = aggregate(pos4cls_Q, cls4lit_K, cls4lit_V, positive_connections)
                cls4neg_aggr_V = aggregate(neg4cls_Q, cls4lit_K, cls4lit_V, negative_connections)

                positive_literal_preembeddings = tf.concat([pos4pos_V, neg4pos_V, cls4pos_aggr_V], axis=-1)
                negative_literal_preembeddings = tf.concat([neg4neg_V, pos4neg_V, cls4neg_aggr_V], axis=-1)

                clause_embeddings = basic_MLP(clause_preembeddings, 'cls_pre2emb', end_activation=EMBED_ACTIVATION)

                positive_literal_embeddings = basic_MLP(positive_literal_preembeddings, 'lit_pre2emb', end_activation=EMBED_ACTIVATION)
                negative_literal_embeddings = basic_MLP(negative_literal_preembeddings, 'lit_pre2emb', end_activation=EMBED_ACTIVATION)

            assert_shape(positive_literal_embeddings, [batch_size, None, EMBEDDING_SIZE])
            assert_shape(negative_literal_embeddings, [batch_size, None, EMBEDDING_SIZE])

            self.positive_policy_logits = tf.keras.layers.Dense(1, name='policy')(positive_literal_embeddings)
            self.negative_policy_logits = tf.keras.layers.Dense(1, name='policy')(negative_literal_embeddings)
            self.policy_logits = tf.concat([self.positive_policy_logits, self.negative_policy_logits], axis=2)
            assert_shape(self.policy_logits, [batch_size, None, 2])

            self.sat_logits = (tf.reduce_sum(tf.keras.layers.Dense(1, name='sat')(positive_literal_embeddings), axis=[1, 2]) +
                               tf.reduce_sum(tf.keras.layers.Dense(1, name='sat')(negative_literal_embeddings), axis=[1, 2]))
            assert_shape(self.sat_logits, [batch_size])

            # zero out policy for test when UNSAT
            # requires sat_labels to be provided, so needs to be a separate tensor in order
            # for inference to work
            self.policy_logits_for_cmp = tf.reshape(self.sat_labels, [batch_size, 1, 1]) * self.policy_logits

            self.policy_loss = tf.keras.losses.binary_crossentropy(self.policy_labels, self.policy_logits_for_cmp, from_logits=True)
            self.policy_probabilities = tf.sigmoid(self.policy_logits, name='policy_prob')
            self.policy_probabilities_for_cmp = tf.sigmoid(self.policy_logits_for_cmp)
            self.policy_weights = tf.reshape(self.sat_labels, [batch_size, 1, 1])
            self.policy_list.append(tf.round(self.policy_probabilities_for_cmp))

            self.sat_loss = tf.keras.losses.binary_crossentropy(self.sat_labels, self.sat_logits, from_logits=True)
            self.sat_probabilities = tf.sigmoid(self.sat_logits, name='sat_prob')
            self.sat_list.append(tf.round(self.sat_probabilities))

            # we do not want to count unsat into policy_error
            self.policy_error = (tf.reduce_sum(tf.abs(tf.round(self.policy_probabilities_for_cmp) - self.policy_labels)) /
                                 tf.reduce_sum(self.sat_labels) /
                                 (tf.cast(variable_num, dtype=tf.float32) * 2.0))
            self.sat_error = tf.reduce_mean(tf.abs(tf.round(self.sat_probabilities) - self.sat_labels))

            self.level_loss = SAT_LOSS_WEIGHT * self.sat_loss + POLICY_LOSS_WEIGHT * self.policy_loss
            self.loss += self.level_loss

            if FLAGS.add_summaries:
                lvln = "_level_{}".format(level)
                tf.summary.scalar("loss" + lvln, self.level_loss)
                tf.summary.scalar("policy_loss" + lvln, self.policy_loss)
                tf.summary.scalar("policy_error" + lvln, self.policy_error)
                tf.summary.scalar("sat_loss" + lvln, self.sat_loss)
                tf.summary.scalar("sat_error" + lvln, self.sat_error)
                tf.summary.scalar("sat_fraction" + lvln, tf.reduce_sum(self.sat_labels) / tf.cast(batch_size, dtype=tf.float32))

        if FLAGS.add_summaries:
            tf.summary.scalar("policy_loss", self.policy_loss)
            tf.summary.scalar("policy_error", self.policy_error)
            tf.summary.scalar("sat_loss", self.sat_loss)
            tf.summary.scalar("sat_error", self.sat_error)
            tf.summary.scalar("sat_fraction", tf.reduce_sum(self.sat_labels) / tf.cast(batch_size, dtype=tf.float32))


def model_fn(features, labels, mode, params):
    del params

    graph = Graph(DEFAULT_SETTINGS, features=features, labels=labels)

    if FLAGS.tpu_enable_host_call:
        host_call = host_call_ported.create_host_call(FLAGS.model_dir)
    else:
        host_call = None
    host_call_ported.remove_summaries()

    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(sat_labels, policy_labels, policy_weights, *args):
            metrics = {}
            for num, (sat, policy) in enumerate(zip(args[0::2], args[1::2])):
                leveln = str(num)
                metrics.update({
                    'sat_error_' + leveln: tf.metrics.mean_absolute_error(labels=sat_labels, predictions=sat),
                    'policy_error_' + leveln: tf.metrics.mean_absolute_error(labels=policy_labels, predictions=policy, weights=policy_weights)
                })
            return metrics

        return tf.estimator.tpu.TPUEstimatorSpec(
            mode, loss=graph.loss, eval_metrics=(metric_fn, [
                graph.sat_labels,
                graph.policy_labels, graph.policy_weights] + [
                x for t in zip(graph.sat_list, graph.policy_list) for x in t
            ]),
            host_call=host_call)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        loss = graph.loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

        if FLAGS.use_tpu:
            optimizer = tf.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

        return tf.estimator.tpu.TPUEstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            host_call=host_call)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={
                'sat_probabilities': graph.sat_probabilities,
                'policy_probabilities': graph.policy_probabilities
            })

    else:
        assert False


def dummy_sample():
    sample_number = FLAGS.batch_size
    variable_number = FLAGS.variable_number
    clause_num = FLAGS.clause_number

    features = np.asarray([[[[1, 0] for _ in range(clause_num)]
                            for _ in range(variable_number)]
                           for _ in range(sample_number)])

    policy_labels = np.asarray([[[1, 0] for _ in range(variable_number)]
                                for _ in range(sample_number)])
    sat_labels = np.asarray([True for _ in range(sample_number)])

    return features, sat_labels, policy_labels


def make_dataset(filename, gzipped):
    batch_size = FLAGS.batch_size
    variable_num = FLAGS.variable_number
    clause_num = FLAGS.clause_number

    def parser(serialized_example):
        return tf.io.parse_single_example(
            serialized_example,
            features={
                'inputs': tf.io.FixedLenFeature([variable_num, clause_num, 2], tf.float32),
                'sat': tf.io.FixedLenFeature([], tf.float32),
                'policy': tf.io.FixedLenFeature([variable_num, 2], tf.float32),
            })

    dataset = tf.data.TFRecordDataset(tf.io.matching_files(filename),
                                      compression_type='GZIP' if gzipped else '')
    dataset = dataset.map(parser, num_parallel_calls=batch_size)
    dataset = dataset.map(lambda x: (x["inputs"], {"sat": x["sat"], "policy": x["policy"]}))

    return dataset


def train_input_fn(params):
    del params

    dataset = make_dataset(FLAGS.train_file, FLAGS.train_files_gzipped)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(4).cache().repeat()
    return dataset


def eval_input_fn(params):
    del params

    dataset = make_dataset(FLAGS.test_file, FLAGS.test_files_gzipped)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    return dataset


def dummy_train_input_fn(params):
    del params

    features, sat_labels, policy_labels = dummy_sample()

    features, labels = (
        features.astype(np.float32),
        {"sat": np.asarray(sat_labels).astype(np.float32),
         "policy": policy_labels.astype(np.float32)})
    ds = tf.data.Dataset.from_tensors((features, labels)).repeat()
    return ds


def serving_input_receiver_fn():
    feature = tf.keras.Input(shape=[None, None, None, 2])

    return tf.estimator.export.ServingInputReceiver(feature, feature)


def main(argv):
    del argv  # Unused.

    if FLAGS.use_tpu:
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            print('Running on TPU ', resolver.master())
        except ValueError:
            raise RuntimeError('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
    else:
        strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        features, labels = next(iter(train_input_fn()))
        graph = Graph(DEFAULT_SETTINGS, features=features, labels=labels)
        model = tf.keras.Model(inputs=[graph.inputs, graph.policy_labels, graph.sat_labels],
                               outputs=[graph.policy_logits, graph.sat_logits])
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
        model.compile(optimizer=optimizer, loss={'loss': lambda y_true, y_pred: y_pred})

    if FLAGS.train_steps > 0:
        model.fit(train_input_fn(), steps_per_epoch=FLAGS.train_steps)

    if FLAGS.export_model:
        tf.saved_model.save(model, FLAGS.export_dir)

    if FLAGS.test_steps > 0:
        model.evaluate(eval_input_fn(), steps=FLAGS.test_steps)


if __name__ == "__main__":
    main(None)
