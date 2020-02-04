import json
import os

import tensorflow as tf
from models.transformer import Transformer
import logging

FORMAT = "%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def define_flags():
    tf.flags.DEFINE_string(
        name="input_dir",
        default="",
        help="Input directory."
    )
    tf.flags.DEFINE_integer(
        name="batch_size",
        default=256,
        help="Bstch size."
    )
    tf.flags.DEFINE_string(
        name="ckpt",
        default="",
        help="Checkpoint directory."
    )
    tf.flags.DEFINE_integer(
        name="epochs",
        default=1,
        help="Epochs."
    )
    tf.flags.DEFINE_string(
        name="work_dir",
        default="",
        help="Workplace."
    )


def main(mode='train'):
    """
    Note: TF_CONFIG is parsed and TensorFlow's GRPC servers are started at the time 
    MultiWorkerMirroredStrategy.init() is called, so TF_CONFIG environment variable
    must be set before a tf.distribute.Strategy instance is created.
    """
    train_datasets = None
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    callbacks = [
        # TODO
        tf.keras.callbacks.TensorBorad(log_dir=''),
        tf.keras.callbacks.ModelCheckpoint(filepath=flags_obj.ckpt)
    ]

    with strategy.scope():
        model = Transformer()
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam()
        )

    if mode.lower() == "train":
        model.fit(
            x=train_datasets,
            epochs=flags_obj.epochs,
            callbacks=callbacks
        )
    elif mode.lower() == "evaluate":
        model.load_weights(tf.train.latest_checkpoint(flags_obj.ckpt))
        model.evaluate()


if __name__ == "__main__":
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["localhost:12345", "localhost:23456"]
        },
        'task': {'type': 'worker', 'index': 0}
    })

    define_flags()
    flags_obj = tf.flags.FLAGS
    tf.app.run(main=main)
