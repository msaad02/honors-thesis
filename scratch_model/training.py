"""
Training script for the transformer model setup in model.py

Nothing out of the ordinary here. Pull in data from dataset.py, pull in the model,
setup the loss and metrics, and train.

Also exports the model parameters and weights to the `save_dir` folder.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # stop showing tensorflow logs...

from model import Transformer  # Model architecture
from dataset import get_datasets  # Dataset processing
import tensorflow as tf
import json

# ---- Model parameters ----
BATCH_SIZE = 64
EPOCHS = 2  # Most likely not optimal, but it works fairly well
NUM_LAYERS = 6  # 4
D_MODEL = 512  # 128
DFF = 2048  # 512
NUM_HEADS = 8  # 8
DROPOUT_RATE = 0.1  # 0.1

save_dir = "./models/transformer_v4/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ---- Pull in the data ----
train_ds, val_ds, text_processor = get_datasets(batch_size=BATCH_SIZE)

# Check GPU is being used. Prints [] if not
physical_devices = tf.config.list_physical_devices("GPU")
print(physical_devices)

# Prevent tensorflow from allocating all GPU memory at once
tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Nice!


# ---- Learning rate schedule ----
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# ---- Setup loss and metrics ----
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


# ---- Training ----
transformer = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    input_vocab_size=5000,  # This is the vocab size used for all the datasets.
    target_vocab_size=5000,
    dropout_rate=DROPOUT_RATE,
)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])

transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# ---- Exporting ----
# We export parameters and weights only to the designated folder
transformer.save_weights(save_dir)

print(transformer.summary())

params = {
    "num_layers": NUM_LAYERS,
    "d_model": D_MODEL,
    "num_heads": NUM_HEADS,
    "dff": DFF,
    "input_vocab_size": 5000,
    "target_vocab_size": 5000,
    "dropout_rate": DROPOUT_RATE,
    "batch_size": BATCH_SIZE,  # To get back text_processor
}

with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

# ---- Export text processor ----
txt_model = tf.keras.Sequential([text_processor])
txt_model.save(save_dir + "text_processor")


print("\nDone!")
