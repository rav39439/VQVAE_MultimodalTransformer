


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ----------------- Vector Quantizer -----------------
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        initializer = tf.random_uniform_initializer()
        self.embeddings = self.add_weight(
            shape=(num_embeddings, embedding_dim),
            initializer=initializer,
            trainable=True,
            name="embeddings"
        )

    def call(self, x, return_indices=False):
        """
        x: (batch, H, W, D)
        return_indices: whether to return the discrete token indices
        """
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # compute distances to embedding
        distances = (
            tf.reduce_sum(flattened**2, axis=1, keepdims=True)
            - 2 * tf.matmul(flattened, self.embeddings, transpose_b=True)
            + tf.reduce_sum(self.embeddings**2, axis=1)
        )

        encoding_indices = tf.argmin(distances, axis=1)  # tokens
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings)
        quantized = tf.reshape(quantized, input_shape)

        # losses
        codebook_loss = tf.reduce_mean((tf.stop_gradient(x) - quantized) ** 2)
        commitment_loss = tf.reduce_mean((x - tf.stop_gradient(quantized)) ** 2)
        self.add_loss(codebook_loss + self.beta * commitment_loss)

        # straight-through estimator
        quantized = x + tf.stop_gradient(quantized - x)

        if return_indices:
            tokens = tf.reshape(encoding_indices, input_shape[:-1])
            return quantized, tokens
        return quantized

vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
vgg.trainable = False

def perceptual_loss(x, y):
    # VGG expects inputs in range [0,255] and BGR ordering
    x_v = tf.keras.applications.vgg16.preprocess_input(x * 255.0)
    y_v = tf.keras.applications.vgg16.preprocess_input(y * 255.0)
    feat_x = vgg(x_v)
    feat_y = vgg(y_v)
    return tf.reduce_mean((feat_x - feat_y) ** 2)


# ----------------- Encoder -----------------
# def get_encoder(latent_dim=256):
#     inputs = keras.Input(shape=(224, 224, 3))
#     x = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(inputs)  # 224->112
#     x = layers.Conv2D(256, 4, strides=2, padding="same", activation="relu")(x)       # 112->56
#     x = layers.Conv2D(512, 4, strides=2, padding="same", activation="relu")(x)       # 56->28
#     x = layers.Conv2D(latent_dim, 1, padding="same")(x)                               # 28x28xlatent_dim
#     return keras.Model(inputs, x, name="encoder")
def get_encoder(latent_dim=512):
    inputs = keras.Input(shape=(224, 224, 3))
    x = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(inputs)   # 224 -> 112
    x = layers.Conv2D(256, 4, strides=2, padding="same", activation="relu")(x)        # 112 -> 56
    
    # No third downsample → MUCH less blur
    x = layers.Conv2D(latent_dim, 1, padding="same")(x)                                # 56×56×latent_dim
    
    return keras.Model(inputs, x, name="encoder")

# ----------------- Decoder -----------------

def get_decoder(latent_dim=512):
    inputs = keras.Input(shape=(56, 56, latent_dim))
    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", activation="relu")(inputs)   # 56 -> 112
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")(x)        # 112 -> 224
    outputs = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="decoder")


# ----------------- VQ-VAE Model -----------------
def get_vqvae(latent_dim=256, num_embeddings=1024):
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    vq_layer = VectorQuantizer(num_embeddings, latent_dim)
    
    inputs = keras.Input(shape=(224, 224, 3))
    latents = encoder(inputs)
    quantized, tokens = vq_layer(latents, return_indices=True)
    outputs = decoder(quantized)
    
    return keras.Model(inputs, [outputs, tokens], name="vq_vae_with_tokens"), encoder, decoder, vq_layer

# ----------------- Decode from Tokens -----------------
def decode_from_tokens(tokens, vq_layer, decoder):
    """
    Reconstruct images purely from token indices.
    tokens: (batch, H, W) integer indices
    vq_layer: trained VectorQuantizer
    decoder: trained decoder model
    """
    batch_size, H, W = tokens.shape
    flat_tokens = tf.reshape(tokens, [-1])
    embeddings = tf.gather(vq_layer.embeddings, flat_tokens)
    quantized_latents = tf.reshape(embeddings, (batch_size, H, W, vq_layer.embedding_dim))
    reconstructed = decoder(quantized_latents)
    return reconstructed

# ----------------- VQVAE Trainer -----------------
class VQVAETrainer(tf.keras.Model):
    def __init__(self, latent_dim=256, num_embeddings=1024, **kwargs):
        super().__init__(**kwargs)
        self.vqvae, self.encoder, self.decoder, self.vq_layer = get_vqvae(latent_dim, num_embeddings)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.vq_loss_tracker]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            recon, _ = self.vqvae(x)
            recon_loss = tf.reduce_mean((x - recon) ** 2)
            total_loss = recon_loss + sum(self.vqvae.losses)

        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }

# ----------------- Example Usage -----------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def show_subplot(original, reconstructed):
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(original, 0, 1))
        plt.title("Original")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(reconstructed, 0, 1))
        plt.title("Reconstructed")
        plt.axis("off")
        plt.show()

    # Initialize trainer
    trainer = VQVAETrainer(latent_dim=256, num_embeddings=1024)
    trainer.compile(optimizer=tf.keras.optimizers.Adam())

    # Dummy images
    x_dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)

#     # Forward pass: get reconstruction + tokens

