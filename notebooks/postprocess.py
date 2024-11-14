import cv2
import numpy as np
import os
 
# Define folder paths
original_folder = 'orig_beigebox'  # Replace with your original images folder path
attacked_folder = 'submission'  # Replace with your attacked images folder path
output_folder = 'output'      # Replace with your output folder path
 
# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
 
# List all images in the attacked folder (we assume filenames are the same in both folders)
attacked_files = os.listdir(attacked_folder)
 
# Process each image pair
for file_name in attacked_files:
    # Check if the file is an image (e.g., .png or .jpg)
    if file_name.endswith(".png") or file_name.endswith(".jpg"):
        original_image_path = os.path.join(original_folder, file_name)  # Path to the original image
        attacked_image_path = os.path.join(attacked_folder, file_name)  # Path to the attacked image
        
        # Ensure both original and attacked images exist
        if os.path.exists(original_image_path) and os.path.exists(attacked_image_path):
            # Open the images
            original_image = cv2.imread(original_image_path)
            attacked_image = cv2.imread(attacked_image_path)
 
            # Convert both images to LAB color space
            original_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
            attacked_lab = cv2.cvtColor(attacked_image, cv2.COLOR_BGR2LAB)
 
            # Split the LAB channels
            l_orig, a_orig, b_orig = cv2.split(original_lab)
            l_att, a_att, b_att = cv2.split(attacked_lab)
 
            # Replace the 'a' and 'b' channels of the attacked image with those from the original image
            a_att = a_orig
            b_att = b_orig
 
            # Merge the new LAB channels (L from attacked, AB from original)
            color_transferred_lab = cv2.merge([l_att, a_att, b_att])
 
            # Convert back to BGR color space
            color_transferred_image = cv2.cvtColor(color_transferred_lab, cv2.COLOR_LAB2BGR)
 
            if sharp: #(not using it)
 
                # Sharpening kernel
                sharpening_kernel = np.array([[-1, -1, -1],
                                              [-1,  9, -1],
                                              [-1, -1, -1]])
            
                # Apply the kernel to the color transferred image
                color_transferred_image = cv2.filter2D(color_transferred_image, -1, sharpening_kernel)
 
            # Save the output image in the output folder with the same filename
            output_image_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_image_path, color_transferred_image)
 
            print(f"Processed and saved: {output_image_path}")