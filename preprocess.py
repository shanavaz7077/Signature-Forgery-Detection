import cv2
import os
from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt

# Specify the folder path containing your images
input_folder_path = '/content/drive/MyDrive/demo/train/orignal'
output_folder_path = '/content/drive/MyDrive/out_put/train/org'

# Get a list of all files in the folder
image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate through each image file
for image_file in image_files:
    # Construct the full path to the input image
    input_image_path = os.path.join(input_folder_path, image_file)

    # Read the input image
    image = cv2.imread(input_image_path)

    # Your existing image processing code
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No contours found in the image {image_file}. The image may not be binary or the signature is not detected.")
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    x = min(box[0] for box in bounding_boxes)
    y = min(box[1] for box in bounding_boxes)
    w = max(box[0] + box[2] for box in bounding_boxes) - x
    h = max(box[1] + box[3] for box in bounding_boxes) - y
    cropped_image = binary_image[y:y+h, x:x+w]

    # Construct the full path to the output image
    output_image_path = os.path.join(output_folder_path, image_file)

    # Save the processed image
    #cv2.imwrite(output_image_path, cropped_image)

    # Optionally, display the processed image
    plt.imshow(cropped_image, cmap="gray")
    plt.show()
