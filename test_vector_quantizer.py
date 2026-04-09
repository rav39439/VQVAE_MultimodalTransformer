import matplotlib.pyplot as plt
import numpy as np
from vector_quantizer_load_train import decoder_new, vq_layer_loaded, vqvae_trainer, x_train

from vector_quantizer import decode_from_tokens

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


# Sample random test image
idx = np.random.choice(len(x_train), 10, replace=False)
test_images = x_train[idx]

# vqvae_model = vqvae_trainer.vqvae  
# reconstructions_test = vqvae_model.predict(test_images, batch_size=4)[0]
# tokens = vqvae_model.predict(test_images, batch_size=4)[1]
# print(type(reconstructions_test))  # likely list
# print(tokens)   # maybe 2 (reconstruction + vq loss)
# # Display images one by one
# for original, reconstructed in zip(test_images, reconstructions_test):
#     # Squeeze in case of extra batch dimension
#     print(reconstructed.shape)
#     print(original.shape)
#     show_subplot(np.squeeze(original), np.squeeze(reconstructed))


#Get the trained VQVAE model
vqvae_model = vqvae_trainer.vqvae

# Get reconstructions AND tokens from the model
recons, tokens = vqvae_model.predict(test_images, batch_size=4)

print("Recon shape:", recons.shape)
print("Token grid shape:", tokens.shape)    # (batch, H, W)
print(tokens)
# Now reconstruct only from tokens
# decoded_from_tokens = decode_from_tokens(tokens, 
#                                          vqvae_trainer.vq_layer, 
#                                          vqvae_trainer.decoder)
decoded_from_tokens = decode_from_tokens(tokens, 
                                         vq_layer_loaded, 
                                         decoder_new)
print("Decoded-from-tokens shape:", decoded_from_tokens.shape)

# Display reconstructions
for i in range(len(test_images)):
    original = test_images[i]
    rec_from_tokens = decoded_from_tokens[i].numpy()

    print("Original:", original.shape)
    print("From tokens:", rec_from_tokens.shape)

    show_subplot(original, rec_from_tokens)
