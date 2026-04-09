import tensorflow as tf
from encoder_decoder_transformer import Transformer, create_look_ahead_mask, create_padding_mask, loss_fn
from parameters import transformer
import numpy as np
# # Assume:
# # tf_dataset_tokens = the tokenized dataset you created
# # transformer = your Transformer model (already built)
# # loss_fn = your loss function for token prediction (e.g., MSE or cross-entropy)
# # create_padding_msk, create_look_ahead_mask = mask functions

# EPOCHS = 5  # or whatever you like
# optimizer = tf.keras.optimizers.Adam(1e-4)

# for epoch in range(EPOCHS):
#     print(f"\nEpoch {epoch+1}/{EPOCHS}")

#     for step, (img_tokens, cap_ids, attn_mask) in enumerate(tf_dataset_tokens):
#         # img_tokens: [batch, 49, 2048]  → decoder target
#         # cap_ids: [batch, MAX_LEN]     → encoder input (text)
#         # attn_mask: [batch, MAX_LEN]   → encoder padding mask
#         print(img_tokens)
#         print(cap_ids)
#         print(attn_mask)

#         with tf.GradientTape() as tape:
#             # Decoder input: image tokens shifted right (teacher forcing)
#             dec_input = img_tokens[:, :-1, :]  # remove last token
#             dec_target = img_tokens[:, 1:, :]  # remove first token

#             # Forward pass
#             predictions, attn_weights = transformer(
#                 inp=cap_ids,
#                 tar=dec_input,
#                 training=True,
#                 enc_padding_mask=create_padding_mask(cap_ids),
#                 look_ahead_mask=create_look_ahead_mask(dec_input.shape[1]),
#                 dec_padding_mask=create_padding_mask(dec_input)
#             )

#             # Compute loss
#             loss = loss_fn(dec_target, predictions)

#         # Backpropagation
#         gradients = tape.gradient(loss, transformer.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

#         print(f"Step {step+1}, Loss: {loss.numpy():.4f}")





# -----------------------------
# Create dummy dataset (10 samples)
# -----------------------------
BATCH_SIZE = 2
NUM_SAMPLES = 10
MAX_LEN = 20
INPUT_VOCAB = 1000
TARGET_VOCAB = 1200

def dummy_gen():
    for _ in range(NUM_SAMPLES):
        inp = np.random.randint(1, INPUT_VOCAB, size=(MAX_LEN,))
        tar = np.random.randint(1, TARGET_VOCAB, size=(MAX_LEN,))

        yield inp, tar

tf_dummy_dataset = tf.data.Dataset.from_generator(
    dummy_gen,
    output_signature=(
        tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
        tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
    )
).batch(BATCH_SIZE)

# -----------------------------
# Create masks
# -----------------------------
def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

# -----------------------------
# Initialize transformer
# -----------------------------
transformer = Transformer(
    num_layers=2,
    d_model=128,
    num_heads=4,
    dff=512,
    input_vocab_size=INPUT_VOCAB,
    target_vocab_size=TARGET_VOCAB,
    pe_input=1000,
    pe_target=1000,
)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)

def loss_fn(real, pred):
    loss_ = loss_object(real, pred)
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    loss_ *= mask
    return tf.reduce_mean(loss_)

optimizer = tf.keras.optimizers.Adam(1e-4)

# -----------------------------
# 🔥 TEST THE MODEL WITH ONE TRAIN STEP
# -----------------------------
for inp, tar in tf_dummy_dataset.take(1):

    tar_inp = tar[:, :-1]       # decoder input
    tar_real = tar[:, 1:]       # what we want model to predict
    print(inp.shape)
    print(tar.shape)

    enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)
    print("mask shapes")
    print(enc_mask.shape)
    print(combined_mask.shape)
    print(dec_mask.shape)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp=inp,
            tar=tar_inp,
            training=True,
            enc_padding_mask=enc_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_mask,
        )

        loss = loss_fn(tar_real, predictions)

    variables = transformer.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    print("Input shape:", inp.shape)
    print("Decoder input shape:", tar_inp.shape)
    print("Predictions shape:", predictions.shape)
    print("Loss:", loss.numpy())

