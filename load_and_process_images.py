import os
import tensorflow as tf
IMAGE_DIR = "/kaggle/input/flickr30k-dataset/flickr30k/Images"
image_paths = [os.path.join(IMAGE_DIR, fname) 
               for fname in os.listdir(IMAGE_DIR) 
               if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_paths)} images in total")

# Take only the first 700 images
image_paths = image_paths[:70]

print(f"Using first {len(image_paths)} images for dataset")



IMG_SIZE = 224

def load_and_preprocess_image(path):
    # Read image from file
    img = tf.io.read_file(path)
    img = tf.io.read_file(path)
    
    # Use TensorFlow string ops to check extension
    is_png = tf.strings.regex_full_match(path, ".*\.png")
    
    # Decode accordingly
    img = tf.cond(is_png,
                  lambda: tf.image.decode_png(img, channels=3),
                  lambda: tf.image.decode_jpeg(img, channels=3))
    
    # Resize
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    
    # Normalize
    img = tf.cast(img, tf.float32) / 255.0
    
    # Set static shape
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img



BATCH_SIZE = 8

dataset = tf.data.Dataset.from_tensor_slices(image_paths)
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(BATCH_SIZE)          # only ONE batch
dataset = dataset.prefetch(tf.data.AUTOTUNE) # prefetch for performance


for batch in dataset.take(1):
    print("Batch shape:", batch.shape)  # shape of the batch: (batch_size, 224, 224, 3)
    
    # Take only the first image in the batch
    one_image = batch[0]
    print("Single image shape:", one_image.shape)  # shape of one image: (224, 224, 3)
    
    # Optional: print min/max values to check normalization
    print("Min pixel value:", tf.reduce_min(one_image).numpy())
    print("Max pixel value:", tf.reduce_max(one_image).numpy())