import tensorflow as tf
import h5py as h5
import numpy as np
import os
import random
import pickle
import time
from datetime import datetime
import sys
from tensorflow_examples.models.pix2pix import pix2pix

def ramp(a, b, t, half_wave=False):
    assert a <= b
    if t <= a:
        return 0.
    elif t >= b:
        return 0. if half_wave else 1.
    else:
        return 0.5 * (1. - np.cos((2. if half_wave else 1.)*np.pi * (t - a)/(b - a)))
ramp = np.vectorize(ramp)

def lin_ramp(x1,y1,x2,y2,t):
    assert x1 <= x2
    if t < x1:
        return y1
    elif t > x2:
        return y2
    return np.interp(t, [x1,x2], [y1,y2])
lin_ramp = np.vectorize(lin_ramp)

LABELLED_BATCH_SIZE = 32
UNLABELLED_BATCH_SIZE = 64
SHUFFLE_SIZE = 512
IMG_SIZE = (128, 128)   # (H, W)
MAX_STEPS = 2048

NUM_VAL_VOLS = 64
NUM_LABELLED_VOLS = 15
MAX_LABELLED_NUM = int(1e10)
MAX_VAL_NUM = 512

LAMBDAS = [
    0.02 * np.concat((
        lin_ramp(0.15, 0.0, 0.5, 1.0, np.linspace(0.0, 0.5, 4)),
        lin_ramp(0.67, 1.0, 1.0, 1.0, np.linspace(0.6, 1.0, 11))
    )),
    np.zeros(15),
]
LAMBDAS = np.array(LAMBDAS[int(sys.argv[1])], dtype='float32')
BLIND_PROB = 0.0

SEED = int(sys.argv[2])

split_num = int(sys.argv[3])

with open('nonempty-paths-vols.pkl', 'rb') as file:
    vols = pickle.load(file)
    
vols.sort()

rng = random.Random(SEED)

rng.shuffle(vols)
split_1 = NUM_VAL_VOLS * split_num
split_2 = NUM_VAL_VOLS * (split_num + 1)
val_vols = vols[split_1:split_2]
train_vols = vols[:split_1] + vols[split_2:]
labelled_vols = train_vols[-NUM_LABELLED_VOLS:]
unlabelled_vols = train_vols[:-NUM_LABELLED_VOLS]

val_paths = [x for vol in val_vols for x in vol]
labelled_paths = [x for vol in labelled_vols for x in vol]
rng.shuffle(labelled_paths)
labelled_paths = labelled_paths[:MAX_LABELLED_NUM]
unlabelled_paths = [x for vol in unlabelled_vols for x in vol]

rng.shuffle(val_paths)
val_paths = val_paths[:MAX_VAL_NUM]
rng.shuffle(labelled_paths)
rng.shuffle(unlabelled_paths)

# THANK YOU CHATGPT FOR WRITING THIS 🙏🙏🙏🙏
# I'M GLAD THE AI UNDERSTANDS APPLE METAL GPU
# PLEASE DON'T TAKE MY JOB THO
def get_dataset(paths, img_size=IMG_SIZE, shuffle_buf=None):
    """H5 → (image, mask) with TF ops; minimal py bridge, stable on Apple Metal.
    If specified (not None), `peekaboo` will randomly hide all channels except one with probability `peekaboo`"""

    # --- tiny py bridge: read raw arrays from .h5 ---
    def _read_h5(path):
        path = path.numpy().decode("utf-8")
        with h5.File(path, "r") as f:
            img = f["image"][:, :, 0:3]                      # type: ignore # (H,W,3)
            msk = f["mask"][...].max(-1)                     # type: ignore # (H,W) foreground>0
            
        img = img.astype("float32")                          # type: ignore # (H,W,3)
        # add channel dim here so TF ops know ranks
        msk = (msk > 0).astype("int32")[..., np.newaxis]     # (H,W,1) 0/1
        return img, msk

    def _py_reader(path):
        img, msk = tf.py_function(_read_h5, [path], [tf.float32, tf.int32]) # type: ignore
        # set static ranks so later ops don’t guess and crash
        img.set_shape([None, None, 3])
        msk.set_shape([None, None, 1])
        return img, msk

    # --- pure TF preprocessing (GPU/Metal friendly) ---
    def _preprocess(img, msk):
        # image: (H,W,1) -> RGB -> resize -> per-image min-max normalize
        img = tf.image.resize(img, img_size)                           # (H,W,3)
        minv = tf.reduce_min(img, axis=[0,1,2])                        # scalar
        maxv = tf.reduce_max(img, axis=[0,1,2])                        # scalar
        img = (img - minv) / (maxv - minv + 1e-8)

        # mask: keep integer class ids, resize w/ nearest, keep channel dim
        msk = tf.image.resize(msk, img_size, method="nearest")         # (h,w,1)
        msk = tf.cast(msk, tf.int32)
        msk = tf.ensure_shape(msk, (img_size[0], img_size[1], 1))
        return img, msk

    opts = tf.data.Options()
    opts.deterministic = False  # let tf parallelize safely

    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
    ds = ds.map(_py_reader, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buf:
        ds = ds.shuffle(shuffle_buf)

    return ds

val_ds = get_dataset(val_paths)
labelled_ds = get_dataset(labelled_paths, shuffle_buf=SHUFFLE_SIZE)
unlabelled_ds = get_dataset(unlabelled_paths, shuffle_buf=SHUFFLE_SIZE)

labelled_batches = (
    labelled_ds
    .cache()
    .repeat()
    .shuffle(SHUFFLE_SIZE)
    .batch(LABELLED_BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

unlabelled_batches = (
    unlabelled_ds
    .cache()
    .repeat()
    .shuffle(SHUFFLE_SIZE)
    .batch(UNLABELLED_BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_batches = (
    val_ds
    .cache()
    .batch(64)
    .prefetch(buffer_size=tf.data.AUTOTUNE))


# base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
base_model = tf.keras.models.load_model( # type: ignore
    "base_model.keras",
    custom_objects=None, compile=True)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs) # type: ignore
# down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3]) # type: ignore

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate() # type: ignore
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose( # type: ignore
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x) # type: ignore

OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)

# Courtesy of ChatGPT
class DiceCoefficient(tf.keras.metrics.Metric): # type: ignore
    def __init__(self, name="dice_coefficient", smooth=1e-6, **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # probs → softmax (multi-class) or sigmoid (binary)
        y_pred = tf.nn.softmax(y_pred, axis=-1)

        # one-hot encode labels if sparse
        if y_true.dtype.is_integer:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1]) # type: ignore

        # flatten
        y_true = tf.reshape(y_true, [-1, tf.shape(y_pred)[-1]]) # type: ignore
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]]) # type: ignore

        intersection = tf.reduce_sum(y_true * y_pred, axis=0)
        union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice = tf.reduce_mean(dice)

        self.dice_sum.assign_add(dice)
        self.count.assign_add(1.0)

    def result(self):
        return self.dice_sum / self.count

    def reset_states(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)

class Logger:
    def __init__(self, filepath, hyperparameters):
        self.filepath = filepath
        self.hyperparameters = hyperparameters
        self.epoch = 0
        with open(self.filepath, "a") as file:
            for var, val in self.hyperparameters.items():
                file.write(f"{var}: {val}\n")
    
    def log_line(self, msg):
        with open(self.filepath, "a") as file:
            file.write(msg + "\n")
    
    def next_epoch(self):
        with open(self.filepath, "a") as file:
            file.write(f"Epoch {self.epoch} started\n")
        self.epoch += 1
    
    def update_training_log(self, step, sec_per_step, stats):
        with open(self.filepath, "a") as file:
            file.write(f"  Step: {step} ({sec_per_step} s/step):\n")
            for stat, val in stats.items():
                file.write(f"    {stat}: {val}\n")

labelled_train_length = len(labelled_paths)
steps_per_epoch = labelled_train_length // LABELLED_BATCH_SIZE

# --- Training Loop + Functionality ---

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) # type: ignore
dice_metric = DiceCoefficient()

@tf.function
def supervised_loss_func(y_true, y_pred):
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred) # type: ignore
    # y_true_one_hot = tf.one_hot(y_true, depth=OUTPUT_CLASSES, axis=3)
    # dice = tf.keras.losses.Dice()(y_true_one_hot, tf.nn.softmax(y_pred))
    
    return cce

# Thx ChatGPT - modified to not use KL divergence loss (no thx ChatGPT)
@tf.function
def unsupervised_loss_func(xu_batch, model, num_channels=3):
    preds = []
    for ch in range(num_channels):
        # mask all but one channel
        mask = tf.one_hot(ch, num_channels)  # shape (num_channels,)
        mask = tf.reshape(mask, (1,1,1,num_channels))
        x_masked = xu_batch * mask  # zero out others

        # forward pass -> logits
        p = model(x_masked, training=True)  # (B,H,W,C)
        # softmax so we’re comparing probs not raw logits
        p = tf.nn.softmax(p, axis=-1)
        preds.append(p)

    preds = tf.stack(preds, axis=0)  # (num_channels, B,H,W,C)

    # average prediction = "reference"
    ref = tf.stop_gradient(tf.reduce_mean(preds, axis=0))  # (B,H,W,C)

    # consistency loss TODO: try MSE, KL (learn what this means), Dice
    loss = 0.
    for p in tf.unstack(preds, axis=0): # type: ignore
        loss += (
            tf.keras.losses.Dice()(ref, p) # type: ignore
        )
    loss /= num_channels
    return loss

# Another chatGPT tensorflow image manipulation magic W
@tf.function
def blind_batch(batch, p):
    batch_size, _, _, channels = batch.shape
    # Randomly choose channel indices for each image
    rand_channels = tf.random.uniform(
        shape=[batch_size], 
        minval=0, 
        maxval=channels, 
        dtype=tf.int32
    )

    one_hot = tf.one_hot(rand_channels, channels, on_value=0., off_value=1.)  # [batch, C]
    channel_mask = tf.reshape(one_hot, [batch_size, 1, 1, channels])
    channel_mask = tf.cast(channel_mask, batch.dtype)

    # Random Bernoulli to decide if we apply masking
    apply_mask = tf.cast(
        tf.random.uniform([batch_size, 1, 1, 1]) < p, 
        batch.dtype
    )
    res = batch * (1 - apply_mask) + batch * channel_mask * apply_mask # type: ignore

    return res

@tf.function
def step(xl_batch, yl_batch, xu_batch, unsupervised_lambda):
    # Calculate loss
    with tf.GradientTape() as tape:
        # xl_blinded = blind_batch(xl_batch, BLIND_PROB)
        yl_pred = model(xl_batch, training=True)
        supervised_loss = supervised_loss_func(yl_batch, yl_pred)
        loss = supervised_loss
        if unsupervised_lambda >= 1e-10:
            unsupervised_loss = unsupervised_loss_func(xu_batch, model)
        else:
            unsupervised_loss = tf.constant(0.0, dtype=tf.float32)
            
        loss += unsupervised_lambda * unsupervised_loss
        
    # Update gradients
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights)) # type: ignore
    
    return supervised_loss, unsupervised_loss

@tf.function
def val_step(x_batch, y_batch, val_metric):
    y = model(x_batch, training=False)
    val_metric.update_state(y_batch, y)
    return supervised_loss_func(y_batch, y)

@tf.function
def update_metric(metric):
    validation_start = time.perf_counter()
    metric.reset_state()
    avg_loss = 0.
    last_loss= -1.
    n = 0.
    for x_batch_val, y_batch_val in val_batches: # type: ignore
        last_loss = val_step(x_batch_val, y_batch_val, metric) # type: ignore
        avg_loss += last_loss # type: ignore
        n += 1.
        
    avg_loss /= n
    metric_val = metric.result()
    
    return (metric_val, {
        "Validation Loss": avg_loss,
        "Last Val Loss": last_loss,
        "DICE Validation": metric_val,
        "Validation Time (s)": time.perf_counter() - validation_start,
        "Num Val Batches": n,
    })

usls = []
sls = []
dice_scores = []

LOG_PATH = 'training-logs'

hyperparams = {
    "LABELLED_BATCH_SIZE": LABELLED_BATCH_SIZE,
    "UNLABELLED_BATCH_SIZE": UNLABELLED_BATCH_SIZE,
    "SHUFFLE_SIZE": SHUFFLE_SIZE,
    "IMG_SIZE": IMG_SIZE,
    "NUM_LABELLED_VOLS": NUM_LABELLED_VOLS,
    "NUM_VAL_VOLS": NUM_VAL_VOLS,
    "MAX_VAL_NUM": MAX_VAL_NUM,
    "LAMBDAS": LAMBDAS,
    "BLIND_PROB": BLIND_PROB,
    "SEED": SEED,
    "LABELLED SLICES": len(labelled_paths),
    "UNLABELLED SLICES": len(unlabelled_paths),
    "VALIDATION SLICES": len(val_paths),
}
logger_path = os.path.join(LOG_PATH, sys.argv[4] + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
os.mkdir(logger_path)
logger = Logger(os.path.join(logger_path,"log.txt"), hyperparams)

# Main training loop
train_start = time.perf_counter()
for epoch, unsupervised_lambda in enumerate(LAMBDAS):
    epoch_start = time.perf_counter()
    logger.next_epoch()
    
    if epoch % 3 == 0:
        model.save_weights(os.path.join(logger_path, f"epoch-{epoch}.weights.h5"))
    
    # Perform training steps
    step_time_sum = 0
    labelled_iter = iter(labelled_batches)
    unlabelled_iter = iter(unlabelled_batches)
    for n_step in range(steps_per_epoch):
        step_start = time.perf_counter()

        (xl_batch, yl_batch) = next(labelled_iter) # type: ignore
        (xu_batch, _) = next(unlabelled_iter) # type: ignore
        [sl, usl] = step(xl_batch, yl_batch, xu_batch, unsupervised_lambda=unsupervised_lambda) # type: ignore
        usls.append(usl)
        sls.append(sl)
        step_time_sum += time.perf_counter() - step_start

        if n_step % 8 == 0:
            [metric_val, stats] = update_metric(dice_metric) # type: ignore
            dice_scores.append(metric_val)
            stats.update({
                "Training Supervised Loss": sl,
                "Training Unsupervised Loss": usl,
                "Unsupervised Lambda": unsupervised_lambda
            })
            logger.update_training_log(n_step, step_time_sum / (n_step+1), stats)

logger.log_line(f"Training done in {time.perf_counter() - train_start:.02f}s.")
logger.log_line(f"Supervised losses: {[x.numpy() for x in sls]}\nUnsupervised losses: {[x.numpy() for x in usls]}")

with open(os.path.join(logger_path, "train-data.pkl"), "wb") as file:
    pickle.dump({
        "hyperparams": hyperparams,
        "dice": dice_scores,
        "sls": sls,
        "usls": usls,
        "lambdas": LAMBDAS,
        "labelled_paths": labelled_paths,
        "unlabelled_paths": unlabelled_paths,
        "seed": SEED,
        "dataset_sizes": {
            "labelled": len(labelled_paths),
            "unlabelled": len(unlabelled_paths),
            "validation": len(val_paths)
        }
    }, file)

tf.keras.backend.clear_session() # type: ignore

os._exit(0)