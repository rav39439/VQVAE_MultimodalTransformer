#-----------dummy file----------------------


# import tensorflow as tf

# # Assuming your VectorQuantzer class is already defined above

# # Instantiate the quantizer
# import tensorflow as tf
# from tensorflow.keras.layers import Layer
# from load_and_process_images import dataset
# import os
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np

# class VectorQuantizer(Layer):
#     def __init__(self, num_emb: int, emb_dim: int, beta: float = 0.25, **kwargs):
#         super().__init__(**kwargs)

#         self.num_emb = num_emb
#         self.emb_dim = emb_dim
#         self.beta = beta

#         self.embeddings = self.add_weight(
#             name="embeddings",
#             shape=(self.emb_dim, self.num_emb),
#             initializer="uniform",
#             trainable=True
#         )

#     def call(self, inputs: tf.Tensor, training: bool = False):
#         original_shape = tf.shape(inputs)
#         flatten_inputs = tf.reshape(inputs, (-1, self.emb_dim))

#         # Get nearest embedding indices
#         encoding_indices = self.__nearest_embedding(flatten_inputs)
#         encodings = tf.one_hot(encoding_indices, self.num_emb)

#         quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
#         quantized = tf.reshape(quantized, original_shape)

#         # Losses
#         commitment_loss = tf.reduce_mean(tf.square(inputs - tf.stop_gradient(quantized)))
#         emb_loss = tf.reduce_mean(tf.square(quantized - tf.stop_gradient(inputs)))
#         self.add_loss(emb_loss + self.beta * commitment_loss)

#         # Straight-through estimator
#         quantized = inputs + tf.stop_gradient(quantized - inputs)

#         return quantized, encoding_indices   # ✔ RETURN BOTH

#     def __nearest_embedding(self, flatten_inputs):
#         distances = (
#             tf.reduce_sum(tf.square(flatten_inputs), axis=1, keepdims=True)
#             + tf.reduce_sum(tf.square(self.embeddings), axis=0)
#             - 2 * tf.matmul(flatten_inputs, self.embeddings)
#         )
#         return tf.argmin(distances, axis=-1)



# num_embeddings = 64
# embedding_dim = 3  # RGB channels
# vq = VectorQuantizer(num_emb=num_embeddings, emb_dim=embedding_dim)
# optimizer = keras.optimizers.Adam(1e-4)

# EPOCHS = 7
# for epoch in range(EPOCHS):
#     print(f"\nEpoch {epoch+1}/{EPOCHS}")

#     for step, batch_images in enumerate(dataset):
#         with tf.GradientTape() as tape:
#             quantized, encoding_indices = vq(batch_images)  

#             # Reconstruction loss
#             recon_loss = tf.reduce_mean(tf.square(batch_images - quantized))

#             # Add VQ internal losses
#             loss = recon_loss + tf.add_n(vq.losses)

#         grads = tape.gradient(loss, vq.trainable_variables)
#         optimizer.apply_gradients(zip(grads, vq.trainable_variables))

#         print(f"Step {step+1}, Loss: {loss.numpy():.4f}")
#         print("Token indices shape:", encoding_indices.shape)

#         # Print first 100 token indices as example
#         print("Token indices sample:", encoding_indices.numpy()[:100])


# # vqvae.save("vqvae_model/")
# # encoder_model.save("vqvae_encoder/")
# # index_model.save("vqvae_indexer/")


# import tensorflow as tf

# # Take a small batch from your dataset
# for batch_images in dataset.take(1):  # first batch
#     print("Original batch shape:", batch_images.shape)

#     # Pass through the trained vector quantizer
#     quantized_batch = vq(batch_images)

#     print("Quantized batch shape:", quantized_batch.shape)

#     # Compare the first image before and after quantization
#     import matplotlib.pyplot as plt

#     original_img = batch_images[0].numpy()
#     quantized_img = quantized_batch[0].numpy()

#     # Plot original and quantized images side by side
#     plt.figure(figsize=(8,4))

#     plt.subplot(1,2,1)
#     plt.title("Original Image")
#     plt.imshow(original_img)
#     plt.axis("off")

#     plt.subplot(1,2,2)
#     plt.title("Quantized Image")
#     plt.imshow(quantized_img)
#     plt.axis("off")

#     plt.show()


# # Take a single batch from image_dataset
# from tensorflow import keras

# # Load the full VQ-VAE model
# vqvae = keras.models.load_model("vqvae_model/", compile=False)

# # Load the encoder (optional, if you want just latents)
# encoder_model = keras.models.load_model("vqvae_encoder/", compile=False)

# # Load the indexer (to get discrete token IDs)
# index_model = keras.models.load_model("vqvae_indexer/", compile=False)

# for batch in image_dataset.take(1):
#     test_images = batch  # shape: (BATCH_SIZE, 224, 224, 3)
#     break

# print("Test batch shape:", test_images.shape)
# # Get reconstructed images
# reconstructed = vqvae.predict(test_images)

# print("Reconstructed images shape:", reconstructed.shape)  # should be same as input: (BATCH_SIZE, 224, 224, 3)


