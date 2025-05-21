import os
from os import listdir
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

data_dir = "Dataset/processed"
output_dir = "Dataset/augmented"

datagen = ImageDataGenerator(
    rotation_range=20,            # Rotate images by 20 degrees randomly
    width_shift_range=0.2,        # Shift images horizontally by 20%
    height_shift_range=0.2,       # Shift images vertically by 20%
    shear_range=0.2,              # Shear images by 20%
    zoom_range=0.2,               # Zoom images by 20%
    horizontal_flip=True,         # Flip images horizontally randomly
    fill_mode='nearest'           # Fill in missing pixels after transformation
)

# Loop through each subfolder class
for folder in listdir(data_dir):
    class_dir = os.path.join(data_dir, folder)

    # Check if it's a directory and not a file
    if os.path.isdir(class_dir):
        # Define output directory for this class
        output_class_dir = os.path.join(output_dir, folder)
        os.makedirs(output_class_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Loop through each image in the class directory
        for image_file in listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)

            # Skip non-image files
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                img = load_img(image_path)  # Load image
                x = img_to_array(img)       # Convert image to numpy array
                x = x.reshape((1,) + x.shape)  # Reshape for ImageDataGenerator

                # Generate 5 augmented images per input image
                i = 0
                for batch in datagen.flow(x, batch_size=1,
                                          save_to_dir=output_class_dir,
                                          save_prefix="aug",
                                          save_format='jpeg'):
                    i += 1
                    if i >= 5:
                        break  # Stop after generating 5 images per original image
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
