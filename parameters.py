
import tensorflow as tf
from encoder_decoder_transformer import Transformer

num_layers = 4
d_model = 256
num_heads = 8
dff = 512
dropout_rate = 0.1
from vector_quantizer_load_train import IMG_SIZE, PATCH_SIZE

MAX_LEN = 32
num_patches = (IMG_SIZE[0] // PATCH_SIZE) * (IMG_SIZE[1] // PATCH_SIZE)
patch_dim = PATCH_SIZE * PATCH_SIZE * 3
target_vocab_size = len(keras_tokenizer.word_index) + 1
patch_projection = tf.keras.layers.Dense(d_model)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

optimizer = tf.keras.optimizers.Adam(1e-4)


# Assuming Encoder and Decoder classes are defined
num_layers = 6
d_model = 128
dff = 512
num_heads = 4
dropout_rate = 0.1
EPOCHS = 50




transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=200,
    target_vocab_size=200,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

