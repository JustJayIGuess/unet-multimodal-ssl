# %% [markdown]
# > Note: 2 runs per split * 5 splits per seed * 5 seeds per ratio * 8 ratios = 400 splits
# 
# ## Plan
# ```python
# for num_labelled in [2, 4, 8, ..., 256]:
#     for seed in [1..5]:
#         for split in [1..5]:
#             run_fsl(num_labelled, seed, split)
#             run_ssl(num_labelled, seed, split)
# ```
# <br/>
# - Each seed gives a different selection of labelled data
# - The five splits are done on the same selection of labelled data

# %% [markdown]
# # Imports

# %%
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import keras
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import nibabel as nib
import random
from sklearn.model_selection import KFold
import datetime

# %% [markdown]
# # Constants

# %%
CROSS_VAL_K = 5
TRAINING_DATA_PATH = "training-data/{}/BraTS20_Training_{:03}{}.nii"

NUM_LABELLED = 16
CROSS_VAL_SPLIT = 0
DATASET_SEED = 0

assert 0 <= CROSS_VAL_SPLIT < CROSS_VAL_K

# %% [markdown]
# # Definitions

# %%
def get_paths_from_id(id: int) -> tuple[str, str]:
    """Get path to inputs volume and segmented volume by case ID.

    Args:
        id (int): ID of the volume, ranging from 1-369.

    Returns:
        tuple[str,str]: A tuple with the inputs path first and the
        segmentation path second
    """    
    return TRAINING_DATA_PATH.format('input', id, ''), TRAINING_DATA_PATH.format('seg', id, '_seg')

def load_nii(path: str) -> npt.NDArray[np.float32]:
    """Load a Nifti file (as `float32`)

    Args:
        path (str): Path to the Nifti file

    Returns:
        npt.NDArray[np.float32]: The loaded `nibabel` object
    """    
    return nib.load(path).get_fdata().astype(np.float32)

def load_volume(
    id: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Load a volume by case ID

    Args:
        id (int): The case ID of the volume

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: The loaded
        volume's inputs (t1, t1ce, t2 as channels), and the segmentation mask
    """    
    inputs_path, seg_path = get_paths_from_id(id)    
    x = load_nii(inputs_path)
    y = load_nii(seg_path) # (H, W, D)
    return x, y

def slice_generator(volume_ids: npt.ArrayLike):
    """Generate slices from volumes with the given IDs

    Args:
        volume_ids (npt.ArrayLike): List of case IDs (integers 1-369)

    Yields:
        tuple[npt.NDArray[np.float32],npt.NDArray[np.float32]]: The loaded
        volumes' inputs (t1, t1ce, t2 as channels) first, and the segmentation
        masks second
    """    
    for id in np.asarray(volume_ids):
        x, y = load_volume(id)  # (H,W,D,4), pre-preprocessed
        for z in range(x.shape[2]):
            yield id, z, x[:, :, z, 1:], y[:, :, z]

def plot_slice(
    input: npt.NDArray[np.float32],
    seg1: npt.NDArray[np.float32],
    seg2: npt.NDArray[np.float32] | None = None,
    title: str | None = None
) -> None:
    """Plot the modalities of a slice from a volume.

    Args:
        input (npt.NDArray[np.float32]): The input modalities, in numpy form.
        seg (npt.NDArray[np.float32]): The segmentation to plot alongside the input.
        title (str, optional): The title of the plot. Defaults to None.
    """    
    orig = input.copy()
    input -= input.min(axis=(0,1))
    input /= input.max(axis=(0,1))
    input[orig == 0.0] = 0.0
    
    cols = 3 if isinstance(seg2, np.ndarray) else 2
    fig, axes = plt.subplots(
        1, cols,
        figsize=(4*cols, 4),
        constrained_layout=True
    )
    
    ax1, ax2 = axes[0:2]
    ax1.imshow(input)
    ax1.axis("off")
    ax1.set_title("Inputs")
    ax2.imshow(seg1)
    ax2.axis("off")
    ax2.set_title("Ground Truth")
    
    ax1.scatter([], [], label="T1", c='r')
    ax1.scatter([], [], label="T1ce", c='g')
    ax1.scatter([], [], label="T2", c='b')
    ax1.legend(loc="lower right", frameon=True)
    
    if cols == 3:
        ax3 = axes[2]
        ax3.imshow(seg2)
        ax3.axis("off")
        ax3.set_title("Prediction")
        
    if title is not None:
        fig.suptitle(title)

def unet_model(output_channels:int) -> keras.Model:
    """Create a U-Net model with pretrained weights and specified number of segmentation classes.

    Args:
        output_channels (int): The number of segmentation classes (i.e., logits in the output layer)

    Returns:
        keras.Model: The U-Net model object
    """    
    base_model: keras.Model = keras.models.load_model(
        "base_model.keras",
        custom_objects=None, compile=True)  # type: ignore

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
    down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    # down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    inputs = keras.layers.Input(shape=[128, 128, 3]) 

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = keras.layers.Conv2DTranspose( 
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)

def get_split(
    num_labelled: int,
    seed: int,
    cross_val_split: int
) -> dict[str, npt.NDArray[np.int32]]:
    """Get the volume IDs for the split with given parameters.
    Deterministic and pseudo-random.

    Args:
        num_labelled (int): The number of labelled volumes.
        seed (int): The seed for shuffling the volumes.
        cross_val_split (int): Which k-fold validation split to run.

    Returns:
        dict[str,npt.NDArray[np.int32]]: Dict containing
        `"labelled", "unlabelled" and "val"` volume IDs
    """    
    rng = random.Random(seed)
    volume_ids = list(range(1, 370))
    rng.shuffle(volume_ids)

    labelled_ids = np.array(volume_ids[:num_labelled], dtype=np.int32)
    remaining_ids = np.array(volume_ids[num_labelled:], dtype=np.int32)
    splits = [
        {
            "labelled": labelled_ids,
            "unlabelled": remaining_ids[train_idx],
            "val": remaining_ids[val_idx]
        }
        for train_idx, val_idx in KFold(n_splits=CROSS_VAL_K).split(remaining_ids)
    ]
    
    split = splits[cross_val_split]
    
    return split

class DiceCoefficient(keras.metrics.Metric):
    """Dice coefficient metric.
    """    
    def __init__(self, name="dice_coef", smooth=1e-6, **kwargs):
        """Create a metric object to track Dice coefficient.

        Args:
            name (str, optional): The name of the metric. Defaults to "dice_coef".
            smooth (_type_, optional): Smoothing to prevent division by zero. Defaults to 1e-6.
        """
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric.

        Args:
            y_true (Tensor): The ground truth, sparse-encoded.
            y_pred (Tensor): The model prediction, still in logit form.
        """        
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        y_true = tf.reshape(y_true, [-1, tf.shape(y_pred)[-1]])
        
        intersection = tf.reduce_sum(y_true * y_pred, axis=0)
        union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
        dice = tf.reduce_mean((2.0 * intersection + self.smooth) / (union + self.smooth))
        
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1.0)
    
    def result(self):
        """Get the metric's value

        Returns:
            Tensor: The metric's value
        """
        return self.dice_sum / self.count

    def reset_state(self):
        """Reset the metric
        """
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)

# %% [markdown]
# # Dataset Preparation

# %%
split = get_split(NUM_LABELLED, DATASET_SEED, CROSS_VAL_SPLIT)
print("\n\n".join([f"{k}: {v}" for k,v in split.items()]))

output_signature = (
    tf.TensorSpec(shape=(), dtype=tf.uint32),
    tf.TensorSpec(shape=(), dtype=tf.uint32),
    tf.TensorSpec(shape=(128, 128, 3), dtype=np.float32),
    tf.TensorSpec(shape=(128, 128), dtype=np.float32)
)

# Data in format: (id, z, input, seg)
labelled_ds = tf.data.Dataset.from_generator(
    lambda: slice_generator(split['labelled']),
    output_signature=output_signature
).shuffle(buffer_size=512)

# Data in format: (id, z, input)
# unlabelled_ds = tf.data.Dataset.from_generator(
#     lambda: slice_generator(split['unlabelled']),
#     output_signature=output_signature
# ).cache().shuffle(buffer_size=512)
# # Forget labels
# unlabelled_ds.map(lambda id, z, input, _: (id, z, input), num_parallel_calls=tf.data.AUTOTUNE)

# Data in format: (id, z, input, seg)
# val_ds = tf.data.Dataset.from_generator(
#     lambda: slice_generator(split['val']),
#     output_signature=output_signature
# ).cache().shuffle(buffer_size=512)


labelled_batches = labelled_ds.batch(32).prefetch(2)
# unlabelled_batches = unlabelled_ds.batch(32).prefetch(tf.data.AUTOTUNE)
# val_batches = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# %% [markdown]
# # Model Instantiation

# %%
model = unet_model(output_channels=2)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
supervised_loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# %% [markdown]
# # Logging

# %%
metrics = {
    "train_loss": keras.metrics.Mean("train_loss", dtype=tf.float32),
    "train_acc": keras.metrics.SparseCategoricalAccuracy("train_acc"),
    "train_dice": DiceCoefficient("train_dice"),
    "val_loss": keras.metrics.Mean("val_loss", dtype=tf.float32),
    "val_acc": keras.metrics.SparseCategoricalAccuracy("val_acc"),
    "val_dice": DiceCoefficient("val_dice"),
}

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = f'logs/gradient_tape/{current_time}/train'
val_log_dir = f'logs/gradient_tape/{current_time}/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

# %% [markdown]
# # Training Definitions

# %%
@tf.function
def step(xl_batch, yl_batch):
    with tf.GradientTape() as tape:
        yl_pred = model(xl_batch, training=True)
        loss = supervised_loss_func(yl_batch, yl_pred)
        
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    # metrics["train_loss"](loss)
    # metrics["train_acc"](yl_batch, yl_pred)
    # metrics["train_dice"](yl_batch, yl_pred)

@tf.function
def val_step(x_batch, y_batch):
    y_pred = model(x_batch, training=False)
    
    # metrics["val_loss"](supervised_loss_func(y_batch, y_pred))
    # metrics["val_acc"](y_batch, y_pred)
    # metrics["val_dice"](y_batch, y_pred)
    

# %% [markdown]
# # Training Loop

# %%
for epoch in range(20):
    for batch_num, (id, z, xl_batch, yl_batch) in enumerate(labelled_batches):
        step(xl_batch, yl_batch)
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', metrics["train_loss"].result(), step=epoch)
        #     tf.summary.scalar('accuracy', metrics["train_acc"].result(), step=epoch)
        #     tf.summary.scalar('dice', metrics["train_dice"].result(), step=epoch)
        
    # for val_batch_num, (vid, vz, vx_batch, vy_batch) in enumerate(val_batches):
    #     if val_batch_num % 8 != 0: continue # only do every eighth slice
    #     val_step(vx_batch, vy_batch)
    #     with val_summary_writer.as_default():
    #         tf.summary.scalar('loss', metrics["val_loss"].result(), step=epoch)
    #         tf.summary.scalar('accuracy', metrics["val_acc"].result(), step=epoch)
    #         tf.summary.scalar('dice', metrics["val_dice"].result(), step=epoch)

    # for metric in metrics.values():
    #     metric.reset_state()

# %% [markdown]
# # Validation

# %%
# for i, (id, z, input, seg1) in enumerate(val_ds.take(5)):
#     pred = model.predict(np.expand_dims(input, axis=0))[0]
#     plot_slice(
#         input.numpy(),
#         seg1.numpy(),
#         pred.argmax(axis=-1),
#         title=f"Volume {id}, z={z}"
#     )


