import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from vector_quantizer import VQVAETrainer, get_vqvae, get_encoder, get_decoder, VectorQuantizer

IMAGE_DIR = "/kaggle/input/flickr30k-dataset/flickr30k/Images"
IMG_SIZE = (224, 224)
MAX_IMAGES = 400
def load_first_n_images(folder, img_size, n):
    imgs = []
    files = sorted(os.listdir(folder))[:n]
    for fname in files:
        img = image.load_img(os.path.join(folder, fname), target_size=img_size)
        img = image.img_to_array(img)
        imgs.append(img)
    return np.array(imgs, dtype=np.float32)

all_images = load_first_n_images(IMAGE_DIR, IMG_SIZE, MAX_IMAGES)
print("Loaded:", all_images.shape)

# Normalize 0–1
all_images = all_images / 255.0

x_train = all_images[:360]
x_test = all_images[360:]

print("Train variance:", np.var(x_train))

# --- Use improved VQ-VAE from previous step -
latent_dim = 512
num_embeddings = 1024

vqvae_trainer = VQVAETrainer(latent_dim=latent_dim, num_embeddings=num_embeddings)
vqvae_trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

# Reduce batch size for larger image
vqvae_trainer.fit(x_train, epochs=10, batch_size=16)
vq_layer=vqvae_trainer.vq_layer

inputs = keras.Input(shape=(56, 56, latent_dim))  # encoder output shape
outputs = vq_layer(inputs, return_indices=False)
vq_model = keras.Model(inputs, outputs, name="vector_quantizer_model")

# Save only weights of this model
vq_weights_path = "/kaggle/working/vector_quantizer.weights.h5"
vq_model.save_weights(vq_weights_path)
print("VectorQuantizer weights saved to:", vq_weights_path)

decoder = vqvae_trainer.decoder
decoder_weights_path = "/kaggle/working/decoder.weights.h5"
decoder.save_weights(decoder_weights_path)


# --- Later: recreate the VQ layer and load weights ---
# 1. Wrap a new layer in a model with same input shape
vq_layer_new = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)
inputs_new = keras.Input(shape=(56, 56, latent_dim))
outputs_new = vq_layer_new(inputs_new, return_indices=False)
vq_model_new = keras.Model(inputs_new, outputs_new, name="vector_quantizer_model_new")

# 2. Dummy forward to build weights
dummy_input = tf.random.normal((1, 56, 56, latent_dim))
_ = vq_model_new(dummy_input)

# 3. Load weights
vq_model_new.load_weights(vq_weights_path)
vq_layer_loaded = vq_model_new.layers[-1]   # last layer in model

decoder_new = get_decoder(latent_dim=latent_dim)
dummy_latents = tf.random.normal((1, 56, 56, latent_dim))
_ = decoder_new(dummy_latents)
decoder_new.load_weights(decoder_weights_path)

print("VectorQuantizer weights loaded successfully!")