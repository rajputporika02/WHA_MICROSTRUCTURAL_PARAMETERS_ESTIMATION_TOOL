import sys
import tkinter as tk
from tkinter import messagebox

def check_requirements():
    if  model_var.get() and micrograph_var.get():
        messagebox.showinfo("Success", "All requirements are met. You can run the code.")
        root.destroy()  # Close the window to proceed
    else:
        messagebox.showerror("Error", "Please make sure all items are checked before proceeding.")
        root.destroy()  # Close the window without running further code
        sys.exit()
# Create the main window
root = tk.Tk()
root.title("Reminder Dialog")
root.geometry("700x300")
root.configure(bg='#2E3440')

# Create variables to hold the checkbox states
model_var = tk.BooleanVar()
micrograph_var = tk.BooleanVar()

# Custom style for checkboxes (thick color tick)
def apply_style(checkbutton):
    checkbutton.configure(selectcolor="#A3BE8C", activebackground="#88C0D0", activeforeground="#ECEFF4", bd=3)

# Create a label with a header
header_label = tk.Label(root, text="Checklist for Micrograph Analysis", font=("Helvetica", 16, "bold"), bg='#2E3440', fg='#88C0D0')
header_label.pack(pady=10)



model_checkbox = tk.Checkbutton(root, text="Does the folder contains model files", variable=model_var, bg='#2E3440', fg='#D8DEE9', font=("Arial", 12), selectcolor="#A3BE8C")
apply_style(model_checkbox)
model_checkbox.pack(anchor=tk.W, padx=20)

micrograph_checkbox = tk.Checkbutton(root, text="Does the folder contain WHA SEM micrograph in .png format and the size of 1024x768 ", variable=micrograph_var, bg='#2E3440', fg='#D8DEE9', font=("Arial", 12), selectcolor="#A3BE8C")
apply_style(micrograph_checkbox)
micrograph_checkbox.pack(anchor=tk.W, padx=20)

# Create a button with some styling
check_button = tk.Button(root, text="Check Requirements", command=check_requirements, bg='#5E81AC', fg='#ECEFF4', font=("Arial", 14, "bold"), relief="raised")
check_button.pack(pady=20)

# Run the application
root.mainloop()






import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Prompt user for input
z = input("Enter the name of the file: ")



# Use the user input to read the correct image
image = cv2.imread(f"{z}.png")

image_size=(image.shape[1],image.shape[0])

if image_size==(1024, 768):
    print("Running...")
    
else:
    sys.exit()
    

# Convert the image to grayscale for thresholding
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the thresholded image using matplotlib
plt.imshow(thresh, cmap='gray')
plt.show()

#cv2.imshow("threshold_image",thresh)
##cv2.waitKey(0)
#cv2.destroyAllWindows()

# Save the thresholded image
plt.imsave(f"{z}_thresh.png", thresh, cmap='gray')





img=cv2.imread(f"{z}_thresh.png",0)
color1=img==0
color2=img==255
allsegment=np.zeros((img.shape[0],img.shape[1],3))
allsegment[color1]=(0,0,0)
allsegment[color2]=(0,1,0)
plt.imsave(f"{z}_thresh_green.png",allsegment)





croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel



max_precipitate_size_um=float(input("Enter the maximum size of a precipitate in (µm): "))
import math
max_precipitate_area_um = math.pi * (max_precipitate_size_um**2) / 4

max_precipitate_size_pixel= max_precipitate_area_um / conversion_factor
import cv2
import numpy as np

def filter_green_clusters(image_path, min_pixels, max_pixels):
    # Step 1: Load the image in color
    image = cv2.imread(image_path)
    
    # Step 2: Create a mask for green pixels (threshold for green)
    # Define the lower and upper bounds for the green color in BGR format
    lower_green = np.array([0, 100, 0])  # Lower bound for green (BGR)
    upper_green = np.array([100, 255, 100])  # Upper bound for green (BGR)

    # Threshold the image to get only green pixels
    green_mask = cv2.inRange(image, lower_green, upper_green)

    # Step 3: Find connected components for green pixels
    num_labels, labels_im = cv2.connectedComponents(green_mask)

    # Create an output image, initially all black
    output_image = np.zeros_like(green_mask)

    # Step 4: Filter the connected components based on the pixel count range
    for label in range(1, num_labels):  # Start from 1 because 0 is the background
        # Get all the pixels that belong to this component
        component = (labels_im == label).astype(np.uint8)

        # Count the number of green pixels in this component
        pixel_size = np.sum(component)

        # Step 5: If the component size is within the specified range, retain it
        if min_pixels <= pixel_size <= max_pixels:
            output_image[component == 1] = 255  # Set pixels of this component to white

    # Step 6: Save and display the output image
    cv2.imwrite(f"{z}_thresh.png", output_image)
    cv2.imshow('Filtered Green Clusters', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
image_path = f"{z}_thresh_green.png"  # Path to the input image
filter_green_clusters(image_path, max_precipitate_size_pixel, 10000000000000000000)








check_condition=int(input("If threshold image is cleaned then enter 1 if not enter 0 : "))
if check_condition==1:
    
    #resize
    img=cv2.imread(f"{z}_thresh.png")
    resize=cv2.resize(img, (512,512))
    cv2.imwrite(f"{z}_thresh_512.png", resize)
    
    #resize
    img=cv2.imread(f"{z}_thresh.png")
    resize=cv2.resize(img, (256,256))
    cv2.imwrite(f"{z}_thresh_256.png", resize)
    
    #prediction    
    import cv2
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
    from keras.utils import normalize
    from matplotlib import pyplot as plt

    # Function to build the U-Net model
    def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = inputs
        
        # Contracting path
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def process_image(model, input_image_path, img_size):
        input_image = cv2.imread(input_image_path, 0)
        input_image_norm = normalize(np.array(input_image), axis=1)
        input_image_resized = cv2.resize(input_image_norm, (img_size, img_size))  # Resize to fit the model input
        input_image_resized = np.expand_dims(np.expand_dims(input_image_resized, axis=0), axis=3)

        # Predict
        prediction = (model.predict(input_image_resized)[0, :, :, 0] > 0.2).astype(np.uint8)
        return prediction

    # Define model and image paths
    model_paths = ['high_mag_e180_256vg.hdf5', 'low_mag_e300_256vg.hdf5']
    input_image_paths = [f"{z}_thresh_256.png", f"{z}_thresh_512.png"]
    img_sizes = [256, 512]

    # Load and predict for all images
    predictions = []
    for model_path, img_size, input_image_path in zip(model_paths * 2, img_sizes * 2, input_image_paths * 2):
        model = simple_unet_model(img_size, img_size, 1)
        model.load_weights(model_path)
        prediction = process_image(model, input_image_path, img_size)
        predictions.append(prediction)

    # Display all predictions with matplotlib
    plt.figure(figsize=(12, 12))
    for i, pred in enumerate(predictions):
        plt.subplot(2, 2, i + 1)
        plt.title(f'Prediction {i+1}')
        plt.imshow(pred, cmap='gray')
    plt.show()

    # Prompt for saving the selected prediction
    save_choice = int(input("Enter the prediction number (1-4) you want to save: ")) - 1
    if 0 <= save_choice < 4:
        cv2.imwrite(f"{z}_thresh_resize_b.png", predictions[save_choice] * 255)
        print(f"Prediction {save_choice + 1} saved successfully.")
    else:
        sys.exit()
    
    
    
    

    
    
    
    
    
    #resize
    img=cv2.imread(f"{z}_thresh_resize_b.png")
    resize=cv2.resize(img, (thresh.shape[1],thresh.shape[0]))
    cv2.imwrite(f"{z}_thresh_resize_b_resize.png", resize)
    
    
    import cv2
    import numpy as np

    image = cv2.imread(f"{z}_thresh_resize_b_resize.png")
    original_image = image.copy()  # Keep a copy of the original image for reset

    # Zoom parameters
    zoom_factor = 1.0
    zoom_step = 0.1
    min_zoom = 0.5
    max_zoom = 3.0

    # Flags and settings
    drawing = False  # True if the mouse is pressed
    erasing = False  # Toggle between drawing and erasing
    brush_size = 2  # Brush size for drawing or erasing
    color = (255, 255, 255)  # Default color (white)
    undo_stack = [image.copy()]  # Stack to store undo history

    # Mouse callback function
    def draw(event, x, y, flags, param):
        global drawing, erasing, image, zoom_factor
        
        # Convert zoomed coordinates back to the original coordinates
        zoom_x = int(x / zoom_factor)
        zoom_y = int(y / zoom_factor)
        
        # Start drawing/erasing when left mouse button is pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            undo_stack.append(image.copy())  # Save the current image state for undo
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        
        # Draw or erase as the mouse moves
        if event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if not erasing:  # Drawing mode
                    cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                else:  # Erasing mode
                    cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

    # Function to apply zoom
    def apply_zoom(image, zoom_factor):
        h, w = image.shape[:2]
        zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
        return zoomed_image

    # Trackbar callback function for brush size
    def set_brush_size(val):
        global brush_size
        brush_size = val

    # Trackbar callback function for color selection
    def set_color(val):
        global color
        if val == 0:
            color = (0, 0, 0)  # Black
        elif val == 1:
            color = (255, 255, 255)  # White

    # Create a window and set mouse callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw)

    # Create trackbars for brush size and color selection
    cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
    cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

    while True:
        # Apply zoom to the image before displaying
        zoomed_image = apply_zoom(image, zoom_factor)
        cv2.imshow('Image', zoomed_image)
        
        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('e'):  # Press 'e' to toggle erasing mode
            erasing = not erasing
            print('Erasing mode:', erasing)
        
        elif key == ord('r'):  # Press 'r' to reset the image
            image = original_image.copy()
            undo_stack.clear()
            undo_stack.append(image.copy())
            print('Image reset')
        
        elif key == ord('q'):  # Press 'q' to quit and save the image
            # Save the image before quitting
            cv2.imwrite(f"{z}_thresh_resize_b_resize.png", image)
            break
        
        elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
            zoom_factor = min(zoom_factor + zoom_step, max_zoom)
            print('Zooming in:', zoom_factor)
        
        elif key == ord('-'):  # Press '-' to zoom out
            zoom_factor = max(zoom_factor - zoom_step, min_zoom)
            print('Zooming out:', zoom_factor)
        
        elif key == ord('u'):  # Press 'u' for undo
            if len(undo_stack) > 1:
                undo_stack.pop()  # Remove the current state
                image = undo_stack[-1].copy()  # Restore the previous state
                print('Undo last action')

    # Close the window
    cv2.destroyAllWindows()
    
    
    #segmentation
    img=cv2.imread(f"{z}_thresh.png",0)
    seg1=img==0
    seg2=img==255
    allsegment=np.zeros((img.shape[0],img.shape[1],3))
    allsegment[seg1]=(1,0,0)
    allsegment[seg2]=(0,1,0)
    plt.imsave(f"{z}_thresh_seg.png",allsegment)
    
    print("The percentage of tungsten is :",(np.sum(seg2==1)*100)/(np.sum(seg2==1)+np.sum(seg2==0)))#true and false 0 ,1
    
    
    
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage import img_as_ubyte

    # Load the image in grayscale mode
    image = cv2.imread(f"{z}_thresh_resize_b_resize.png", cv2.IMREAD_GRAYSCALE)

    # Ensure the white boundary is represented as 255 and the background as 0
    # If your white boundary is not pure white (255), you may need to threshold it
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological erosion to thin the boundary
    # The kernel defines the neighborhood size for erosion, adjust if necessary
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=0)

    # Skeletonize the image to ensure the boundary is thinned
    skeleton = skeletonize(eroded_image // 255)  # Skeletonize expects a binary image (0, 1)
    skeleton_image = img_as_ubyte(skeleton)  # Convert back to 0-255 range

    # Dilate the skeleton to ensure uniform 2-pixel thickness boundary
    thinned_image = cv2.dilate(skeleton_image, kernel, iterations=0)

    # Save the output image

    # Display the result (optional)
    cv2.imwrite(f"{z}_thresh_resize_b_resize_thin.png", thinned_image)



    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_thresh_seg.png")
    boundaryImg = cv2.imread(f"{z}_thresh_resize_b_resize_thin.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [255, 255, 255]
    # Display the merged image


    cv2.imwrite(f"{z}_merged_thin.png", outputImg)





    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_merged_thin.png")

    # Define pixel values for green, white, and black (background)
    green_value = [0, 255, 0]  # Adjust if needed (BGR format)
    white_value = [255, 255, 255]  # White for boundary
    black_value = [0, 0, 0]  # Black for background

    # Convert the image to grayscale for easier processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask for white pixels
    white_mask = cv2.inRange(image, np.array(white_value), np.array(white_value))

    # Create a mask for green pixels
    green_mask = cv2.inRange(image, np.array(green_value), np.array(green_value))

    # Create an output image initialized as black
    output_image = np.zeros_like(gray_image)

    # Get the height and width of the image
    height, width = gray_image.shape

    # Define a function to check if all 8 pixels are green in a specific direction
    def check_green_complete(image, x, y, direction):
        if direction == 'vertical':
            # Check 8 pixels above and 8 pixels below
            if y - 7 >= 0 and y + 7 < height:
                return np.all(image[y-7:y, x] == 255) and np.all(image[y+1:y+8, x] == 255)
        elif direction == 'horizontal':
            # Check 8 pixels left and 8 pixels right
            if x - 7 >= 0 and x + 7 < width:
                return np.all(image[y, x-7:x] == 255) and np.all(image[y, x+1:x+7] == 255)
        return False

    # Iterate over each white pixel
    for y in range(height):
        for x in range(width):
            if white_mask[y, x] == 255:  # If the pixel is white
                # Check vertical and horizontal conditions
                if check_green_complete(green_mask, x, y, 'vertical') or check_green_complete(green_mask, x, y, 'horizontal'):
                    # Set the output pixel to white if conditions are satisfied
                    output_image[y, x] = 255

    # Save the output image
    cv2.imwrite(f"{z}_merged_thin_ex.png", output_image)



    import cv2
    import numpy as np
    from skimage.morphology import disk, dilation
    from skimage import io, img_as_ubyte
    from skimage.color import rgb2gray
    from skimage.feature import canny
    
    # Read the original image
    original_image = cv2.imread(f"{z}_thresh_seg.png")
    
    # Convert to grayscale
    gray_image = rgb2gray(original_image)
    
    # Detect edges using the Canny method
    edge_image = canny(gray_image)
    
    # Define a structuring element for dilation (disk of radius 0)
    selem = disk(0)
    
    # Dilate the edges
    dilated_edges = dilation(edge_image, selem)
    
    # Convert the dilated edge image to uint8 type for proper overlay
    dilated_edges_uint8 = img_as_ubyte(dilated_edges)
    
    # Overlay the dilated edges on the original image (in white)
    overlayed_image = original_image.copy()
    overlayed_image[dilated_edges_uint8 > 0] = [255, 255, 255]
    
    # Display the dilated edges
    
    
    # Save the result image
    cv2.imwrite(f"{z}_perimeter.png", overlayed_image)






    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_perimeter.png")
    boundaryImg = cv2.imread(f"{z}_merged_thin_ex.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [255, 255, 255]
    # Display the merged image





    cv2.imwrite(f"{z}_merged_thin_img.png", outputImg)
    
    
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage import img_as_ubyte

    # Load the image in grayscale mode
    image = cv2.imread(f"{z}_thresh_resize_b_resize.png", cv2.IMREAD_GRAYSCALE)

    # Ensure the white boundary is represented as 255 and the background as 0
    # If your white boundary is not pure white (255), you may need to threshold it
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological erosion to thin the boundary
    # The kernel defines the neighborhood size for erosion, adjust if necessary
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=0)

    # Skeletonize the image to ensure the boundary is thinned
    skeleton = skeletonize(eroded_image // 255)  # Skeletonize expects a binary image (0, 1)
    skeleton_image = img_as_ubyte(skeleton)  # Convert back to 0-255 range

    # Dilate the skeleton to ensure uniform 2-pixel thickness boundary
    thinned_image = cv2.dilate(skeleton_image, kernel, iterations=1)

    # Save the output image

    # Display the result (optional)
    #cv2.imshow('Thinned Boundary Image', thinned_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    cv2.imwrite(f"{z}_thresh_resize_b_resize_thick2pix.png", thinned_image)



    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_thresh_seg.png")
    boundaryImg = cv2.imread(f"{z}_thresh_resize_b_resize_thick2pix.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [255, 255, 255]
    # Display the merged image

    #cv2.imshow('Merged Image', outputImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save the merged image if needed
    cv2.imwrite(f"{z}_merge_thick2pix.png", outputImg)
    
    
    
    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_thresh_seg.png")
    boundaryImg = cv2.imread(f"{z}_thresh_resize_b_resize_thick2pix.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [0, 0, 0]
    # Display the merged image

    #cv2.imshow('Merged Image', outputImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save the merged image if needed
    cv2.imwrite(f"{z}_merge_thick2pix_black_b.png", outputImg)
    
    
    
    import cv2
    import numpy as np

    #image = cv2.imread(f"{z}_merge_thick2pix.png")
    #original_image = image.copy()  # Keep a copy of the original image for reset

    # Zoom parameters
    #zoom_factor = 1.0
    #zoom_step = 0.1
   # min_zoom = 0.5
   #max_zoom = 3.0

    # Flags and settings
    #drawing = False  # True if the mouse is pressed
    #erasing = False  # Toggle between drawing and erasing
    #brush_size = 2  # Brush size for drawing or erasing
    #color = (255, 255, 255)  # Default color (white)
    #undo_stack = [image.copy()]  # Stack to store undo history

    # Mouse callback function
    #def draw(event, x, y, flags, param):
        #global drawing, erasing, image, zoom_factor
        
        # Convert zoomed coordinates back to the original coordinates
        #zoom_x = int(x / zoom_factor)
        #zoom_y = int(y / zoom_factor)
        
        # Start drawing/erasing when left mouse button is pressed
        #if event == cv2.EVENT_LBUTTONDOWN:
            #drawing = True
            #undo_stack.append(image.copy())  # Save the current image state for undo
        #elif event == cv2.EVENT_LBUTTONUP:
           # drawing = False
        
        # Draw or erase as the mouse moves
        #if event == cv2.EVENT_MOUSEMOVE:
            #if drawing:
                #if not erasing:  # Drawing mode
                    #cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                #else:  # Erasing mode
                   # cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

    # Function to apply zoom
    #def apply_zoom(image, zoom_factor):
        #h, w = image.shape[:2]
        #zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
        #return zoomed_image

    # Trackbar callback function for brush size
    #def set_brush_size(val):
        #global brush_size
        #brush_size = val

    # Trackbar callback function for color selection
    #def set_color(val):
        #global color
       # if val == 0:
           # color = (255, 0, 0)  # Blue
       # elif val == 1:
            #color = (255, 255, 255)  # White

    # Create a window and set mouse callback
    #cv2.namedWindow('Image')
    #cv2.setMouseCallback('Image', draw)

    # Create trackbars for brush size and color selection
    #cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
    #cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

    #while True:
        # Apply zoom to the image before displaying
        #zoomed_image = apply_zoom(image, zoom_factor)
        #cv2.imshow('Image', zoomed_image)
        
        # Wait for keypress
        #key = cv2.waitKey(1) & 0xFF
        
        #if key == ord('e'):  # Press 'e' to toggle erasing mode
            #erasing = not erasing
            #print('Erasing mode:', erasing)
        
        #elif key == ord('r'):  # Press 'r' to reset the image
            #image = original_image.copy()
           # undo_stack.clear()
           # undo_stack.append(image.copy())
            #print('Image reset')
        
        #elif key == ord('q'):  # Press 'q' to quit and save the image
            # Save the image before quitting
            #cv2.imwrite(f"{z}_merge_thick2pix.png", image)
           # break
        
        #elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
            #zoom_factor = min(zoom_factor + zoom_step, max_zoom)
            #print('Zooming in:', zoom_factor)
        
        #elif key == ord('-'):  # Press '-' to zoom out
            #zoom_factor = max(zoom_factor - zoom_step, min_zoom)
           # print('Zooming out:', zoom_factor)
        
        #elif key == ord('u'):  # Press 'u' for undo
            #if len(undo_stack) > 1:
                #undo_stack.pop()  # Remove the current state
               # image = undo_stack[-1].copy()  # Restore the previous state
               # print('Undo last action')

    # Close the window
    #cv2.destroyAllWindows()

    
    
    
    
    
    import cv2
    import numpy as np

    # Step 1: Load the image
    image = cv2.imread(f"{z}_merge_thick2pix.png")

    # Step 2: Separate the channels
    green_channel = image[:, :, 1]  # Extract the green channel
    red_channel = image[:, :, 2]  # Extract the red channel

    # Step 3: Create a mask for the white boundary
    lower_white = np.array([250, 250, 250])  # Lower bound for white
    upper_white = np.array([255, 255, 255])  # Upper bound for white
    white_mask = cv2.inRange(image, lower_white, upper_white)

    # Step 4: Find contours of the white boundary
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Create a mask for enclosed green regions
    green_enclosed_mask = np.zeros_like(green_channel, dtype=np.uint8)

    for contour in contours:
        # Create a blank mask for each contour
        contour_mask = np.zeros_like(green_channel, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)  # Fill the contour
        
        # Check if green pixels are within the contour
        green_in_contour = cv2.bitwise_and(green_channel, contour_mask)
        if np.any(green_in_contour > 0):  # If green is present inside the contour
            green_enclosed_mask = cv2.bitwise_or(green_enclosed_mask, green_in_contour)

    # Step 6: Combine the green enclosed regions with the white boundary
    output_image = np.zeros_like(image)  # Start with a black image
    output_image[:, :, 1] = green_enclosed_mask  # Add the enclosed green regions
    output_image[:, :, :] = cv2.bitwise_or(output_image, cv2.merge([white_mask, white_mask, white_mask]))  # Add the white boundary

    # Step 7: Save or display the result
    cv2.imwrite(f"{z}_enclosed_green_with_boundary.png", output_image)
    #cv2.imshow('Enclosed Green Regions with White Boundary', output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_enclosed_green_with_boundary.png")

    # Convert the image from BGR to HSV color space for easier color thresholding
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the green color in HSV
    lower_green = np.array([35, 100, 100])  # Adjust these values based on your image
    upper_green = np.array([85, 255, 255])

    # Create a mask for green pixels
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Create an output image where everything else is black
    output = np.zeros_like(image)
    output[mask > 0] = image[mask > 0]

    # Convert the non-green pixels to black
    output[mask == 0] = [0, 0, 0]  # Background becomes black

    # Save and display the result
    cv2.imwrite(f"{z}_enclosed_green_with_boundary_ex_green.png", output)
    #cv2.imshow('Green Pixels', output)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



    import cv2
    import numpy as np

    def filter_green_clusters(image_path, min_pixels, max_pixels):
        # Step 1: Load the image in color
        image = cv2.imread(image_path)
        
        # Step 2: Create a mask for green pixels (threshold for green)
        # Define the lower and upper bounds for the green color in BGR format
        lower_green = np.array([0, 100, 0])  # Lower bound for green (BGR)
        upper_green = np.array([100, 255, 100])  # Upper bound for green (BGR)

        # Threshold the image to get only green pixels
        green_mask = cv2.inRange(image, lower_green, upper_green)

        # Step 3: Find connected components for green pixels
        num_labels, labels_im = cv2.connectedComponents(green_mask)

        # Create an output image, initially all black
        output_image = np.zeros_like(green_mask)

        # Step 4: Filter the connected components based on the pixel count range
        for label in range(1, num_labels):  # Start from 1 because 0 is the background
            # Get all the pixels that belong to this component
            component = (labels_im == label).astype(np.uint8)

            # Count the number of green pixels in this component
            pixel_size = np.sum(component)

            # Step 5: If the component size is within the specified range, retain it
            if min_pixels <= pixel_size <= max_pixels:
                output_image[component == 1] = 255  # Set pixels of this component to white

        # Step 6: Save and display the output image
        cv2.imwrite(f"{z}_enclosed_binary.png", output_image)
        #cv2.imshow('Filtered Green Clusters', output_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    # Usage example
    image_path = f"{z}_enclosed_green_with_boundary_ex_green.png"  # Path to the input image
    filter_green_clusters(image_path, 7, 1000000000)



    import matplotlib.pyplot as plt

    img=cv2.imread(f"{z}_enclosed_binary.png",0)
    seg_1=img==0
    seg_2=img==255
    allsegment=np.zeros((img.shape[0],img.shape[1],3))
    allsegment[seg_1]=(1,0,0)
    allsegment[seg_2]=(0,1,0)
    plt.imsave(f"{z}_enclosed_binary_seg.png",allsegment)




    






    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Load the image
    image = cv2.imread(f"{z}_enclosed_binary_seg.png")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Threshold the image to isolate green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours of the green regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image for drawing
    output_image = image.copy()

    # Get scale information
    #croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
    #croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
    #conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel

    # Initialize variables
    valid_contours = 0
    total_major_diameter = 0
    contour_data = []
    major_diameters = []

    # Process contours
    for i, contour in enumerate(contours, 1):
        if cv2.contourArea(contour) < 1:  # Skip small contours
            continue

        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cX = int(moments['m10'] / moments['m00'])
            cY = int(moments['m01'] / moments['m00'])

            # Bounding box and major diameter
            x, y, w, h = cv2.boundingRect(contour)
            major_diameter = max(w, h) * conversion_factor  # µm
            area_um2 = cv2.contourArea(contour) * conversion_factor ** 2
            perimeter_um = cv2.arcLength(contour, True) * conversion_factor
            particle_edge_um = area_um2 ** 0.5
            circularity_um = 4 * 3.14 * area_um2 / (perimeter_um ** 2)

            # Update variables
            total_major_diameter += major_diameter
            major_diameters.append(major_diameter)
            contour_data.append((contour, cX, cY, major_diameter))
            valid_contours += 1

            # Output details
            print(f'Region {valid_contours}:')
            print(f'  Area: {area_um2:.2f} µm²')
            print(f'  Perimeter: {perimeter_um:.2f} µm')
            print(f'  Major Diameter: {major_diameter:.2f} µm')
            print(f'  Particle Edge: {particle_edge_um:.2f} µm')
            print(f'  Circularity: {circularity_um:.2f} µm')
            print()

    # Final calculations
    if valid_contours > 0:
        average_major_diameter = total_major_diameter / valid_contours
        min_diameter = min(major_diameters)
        max_diameter = max(major_diameters)
        # Calculate the standard deviation using numpy
        std_dev = np.std(major_diameters)
        print(f"\nAverage Major Diameter: {average_major_diameter:.2f} µm")
        print(f"Minimum Major Diameter: {min(major_diameters):.2f} µm")
        print(f"Maximum Major Diameter: {max(major_diameters):.2f} µm")
        print(f"Standard Deviation: {np.std(major_diameters):.2f} µm")

    # Draw contours and label numbers
    for i, (contour, cX, cY, major_diameter) in enumerate(contour_data, 1):
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        text = str(i)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        textX = cX - text_size[0] // 2
        textY = cY + text_size[1] // 2
        cv2.putText(output_image, text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save and display results
    cv2.imwrite(f"{z}_grain_regions1.png", output_image)
    #cv2.imshow("Labeled Image", output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Particle size distribution
    if major_diameters:
        major_diameters = np.array(major_diameters)
        hist, bins = np.histogram(major_diameters, bins='auto')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        mu, std = norm.fit(major_diameters)

        plt.bar(bin_centers, hist, width=(bins[1] - bins[0]), alpha=0.6, color='purple', edgecolor='black', linewidth=1.0)
        x = np.linspace(bin_centers[0], bin_centers[-1], 100)
        p = norm.pdf(x, mu, std) * len(major_diameters) * (bins[1] - bins[0])
        plt.plot(x, p, 'b', linewidth=2)
        plt.title('W Particle Size Distribution')
        plt.xlabel('W Particle Size (µm)')
        plt.ylabel('Count')
        plt.xlim([0, max(major_diameters) + 10])
        plt.ylim([0, max(hist) + 5])
        plt.savefig(f"{z}_Particle_Size_distribution1.png", dpi=300, bbox_inches='tight')
        plt.show()




    import pandas as pd

    # Initialize a list to store the details of each valid region
    region_details = []
    valid_contours = 0  # Counter for valid regions

    # Process each contour and store the details
    for contour in contours:
        if cv2.contourArea(contour) < 1:  # Ignore small contours
            continue

        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            x, y, w, h = cv2.boundingRect(contour)
            major_diameter = max(w, h) * conversion_factor  # Major diameter in µm
            area_um2 = cv2.contourArea(contour) * conversion_factor ** 2
            perimeter_um = cv2.arcLength(contour, True) * conversion_factor
            particle_edge_um = area_um2 ** 0.5  # Corrected to compute the square root of area_um2
            circularity_um = 4 * 3.14 * area_um2 / (perimeter_um ** 2)  # Corrected to square the perimeter_um variable

            # Increment valid region counter
            valid_contours += 1

            # Append details to the list
            region_details.append({
                'Region': valid_contours,  # Use valid_contours as region index
                'Area (µm²)': round(area_um2, 2),
                'Perimeter (µm)': round(perimeter_um, 2),
                'Major Diameter (µm)': round(major_diameter, 2),
                'Particle Edge (µm)': round(particle_edge_um, 2),
                'Circularity (µm)': round(circularity_um, 2)
            })

    # Ensure all valid regions match the image labels
    if len(region_details) != valid_contours:
        raise ValueError("Mismatch between the number of labeled regions and saved region details.")

    # Create a pandas DataFrame from the region details
    df = pd.DataFrame(region_details)

    # Save the DataFrame to an Excel file
    output_file = f"{z}_region_details.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Region details successfully saved to {output_file}")




    import pandas as pd
    import numpy as np
    from openpyxl import load_workbook

    # Load the Excel file
    file_name = f"{z}_region_details.xlsx"
    sheet_name = "Sheet1"  # Adjust if the sheet has a different name
    data = pd.read_excel(file_name, sheet_name=sheet_name)

    # Assuming the angles are in a column named 'Angles'
    grain_data = data['Major Diameter (µm)']

    # Calculate the average and standard deviation
    average_grain_size22 = np.mean(grain_data)
    std_dev22= np.std(grain_data)

    # Load the workbook and select the sheet
    wb = load_workbook(file_name)
    sheet = wb[sheet_name]

    # Write the results to the Excel file (e.g., in cells D1 and D2)
    sheet["H1"] = "Average Grain Size (µm)"
    sheet["I1"] = f"{average_grain_size22:.2f} "
    sheet["H2"] = "Standard Deviation (µm)"
    sheet["I2"] = f"{std_dev22:.2f} "
    sheet["H3"] = "Min diameter (µm)"
    sheet["I3"] = f"{min_diameter:.2f} "
    sheet["H4"] = "Max diameter (µm)"
    sheet["I4"] = f"{max_diameter:.2f} "

    # Save the updated workbook
    wb.save(file_name)


    ##########grain neck analysis #############################
    #import cv2
    #import numpy as np

    # Load the image
    #image = cv2.imread('2_enclosed_binary_seg.png')

    # Define color ranges
    #green = np.array([0, 255, 0])
    #red = np.array([0, 0, 255])
    #yellow = np.array([0, 255, 255])

    # Get the dimensions of the image
    #rows, cols, _ = image.shape

    # Create a copy of the image to modify
    #output_image = image.copy()

    # Define the directions for checking (horizontal, vertical, diagonal)
    #directions = [
        #(0, 1),  # Horizontal
        #(1, 0),  # Vertical
        #(1, 1),  # Diagonal down-right
       # (1, -1)  # Diagonal down-left
    #]

    # Iterate over the image
    #for row in range(rows):
        #for col in range(cols):
            # Check for the pattern in each direction
            #for dr, dc in directions:
                #for red_count in range(1, 4):  # Check for 1, 2, or 3 red pixels
                    #if (0 <= row + (red_count + 1) * dr < rows) and (0 <= col + (red_count + 1) * dc < cols):
                        #if (np.array_equal(image[row, col], green) and
                            #all(np.array_equal(image[row + i * dr, col + i * dc], red) for i in range(1, red_count + 1)) and
                            #np.array_equal(image[row + (red_count + 1) * dr, col + (red_count + 1) * dc], green)):

                            # Fill the red pixels with yellow and ensure continuity
                            #for i in range(1, red_count + 1):
                                #for j in range(-1, 2):  # To make the thickness 3 pixels
                                    #if 0 <= row + i * dr + j * dc < rows and 0 <= col + i * dc + j * dr < cols:
                                        #output_image[row + i * dr + j * dc, col + i * dc + j * dr] = yellow

    # Save and display the modified image

    #cv2.imshow('Modified Image', output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #cv2.imwrite("one_pixel_yel.png", output_image)







    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_enclosed_binary_seg.png")

    # Define color ranges
    green = np.array([0, 255, 0])
    red = np.array([0, 0, 255])
    yellow = np.array([0, 255, 255])

    # Get the dimensions of the image
    rows, cols, _ = image.shape

    # Create a copy of the image to modify
    output_image = image.copy()

    # Define the directions for checking (horizontal, vertical, diagonal)
    directions = [
        (0, 1),  # Horizontal
        (1, 0),  # Vertical
        (1, 1),  # Diagonal down-right
        (1, -1)  # Diagonal down-left
    ]

    # Iterate over the image
    for row in range(rows):
        for col in range(cols):
            # Check for the pattern in each direction
            for dr, dc in directions:
                for red_count in range(1, 4):  # Check for 1, 2, or 3 red pixels
                    if (0 <= row + (red_count + 1) * dr < rows) and (0 <= col + (red_count + 1) * dc < cols):
                        if (np.array_equal(image[row, col], green) and
                            all(np.array_equal(image[row + i * dr, col + i * dc], red) for i in range(1, red_count + 1)) and
                            np.array_equal(image[row + (red_count + 1) * dr, col + (red_count + 1) * dc], green)):

                            # Fill the red pixels with yellow and ensure continuity
                            for i in range(1, red_count + 1):
                                for j in range(-1, 2):  # To make the thickness 3 pixels
                                    if 0 <= row + i * dr + j * dc < rows and 0 <= col + i * dc + j * dr < cols:
                                        output_image[row + i * dr + j * dc, col + i * dc + j * dr] = yellow

    # Convert yellow to a binary mask
    yellow_mask = cv2.inRange(output_image, yellow, yellow)

    # Find contours of the yellow regions
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and remove thin yellow regions
    for contour in contours:
        if cv2.contourArea(contour) <= 3:  # Removing small yellow areas (1-pixel thick)
            cv2.drawContours(output_image, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Save and display the modified image
    #cv2.imshow('Modified Image', output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite(f"{z}_grains_neck_yellow_black.png", output_image)






    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_grains_neck_yellow_black.png")

    # Define the black color and the new green color
    black = np.array([0, 0, 0])
    green = np.array([0, 255, 0])

    # Replace black pixels with green pixels
    image[np.all(image == black, axis=-1)] = green

    # Save and display the modified image
    #cv2.imshow('Modified Image with Green Pixels', image)
    ##cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite(f"{z}_grains_neck.png", image)






    import cv2
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.stats import norm

    # Load the image
    #image = cv2.imread(f"{z}_grains_neck_yellow_black.png")

    # Convert the image to HSV color space
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV (for detecting grains)
    #lower_green = np.array([40, 40, 40])
    #upper_green = np.array([70, 255, 255])

    # Define the range for yellow color in HSV (for detecting necks)
    #lower_yellow = np.array([20, 100, 100])
    #upper_yellow = np.array([40, 255, 255])

    # Threshold the image to isolate green (grain) and yellow (neck) regions
    #mask_green = cv2.inRange(hsv, lower_green, upper_green)
    #mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the green (grain) regions
    #contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get scale information
    #croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
    #croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
    #conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel

    # Initialize variables
    #valid_contours = 0
    #contour_data = []

    # Kernel for morphological dilation
    #kernel = np.ones((3, 3), np.uint8)

    # Process contours
    #for i, contour in enumerate(contours, 1):
        #if cv2.contourArea(contour) < 1:  # Skip small contours
            #continue

        # Create a mask for the current grain
        #grain_mask = np.zeros(mask_green.shape, dtype=np.uint8)
        #cv2.drawContours(grain_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Dilate the grain mask to find adjacent yellow pixels
        #dilated_grain_mask = cv2.dilate(grain_mask, kernel, iterations=1)
        #yellow_necks = cv2.bitwise_and(dilated_grain_mask, mask_yellow)

        # Count yellow pixels directly in contact with the grain
        #neck_pixel_count = cv2.countNonZero(yellow_necks)

        # Calculate the total length of necks in µm
        #total_neck_length = neck_pixel_count * conversion_factor

        # Calculate the perimeter of the grain in µm
        #perimeter = cv2.arcLength(contour, True) * conversion_factor

        # Calculate contiguity
        #contiguity = total_neck_length / perimeter if perimeter != 0 else 0

        # Update valid contours
        #valid_contours += 1
        #contour_data.append((i, total_neck_length, perimeter, contiguity))

        # Output details for each grain
        #print(f"Region {i}:")
        #print(f"  Total Neck Length: {total_neck_length:.2f} µm")
        #print(f"  Perimeter: {perimeter:.2f} µm")
        #print(f"  Contiguity: {contiguity:.4f}")
        #print()

    # Save output to Excel
    #df = pd.DataFrame(contour_data, columns=['Region', 'Total Neck Length (µm)', 'Perimeter (µm)', 'Contiguity'])
    #df.to_excel(f"{z}_enclosed_grain_neck_contiguty_details.xlsx", index=False)

    # Final output for all grains
    #if valid_contours == 0:
       # print("No valid contours found.")
    #else:
        #print("\nSummary of Contiguities:")
        #for data in contour_data:
            #region, neck_length, perimeter, contiguity = data
            #print(f"Region {region}: Total Neck Length = {neck_length:.2f} µm, Perimeter = {perimeter:.2f} µm, Contiguity = {contiguity:.4f}")
            
    
            
    
    import cv2
    import numpy as np

    # Step 1: Load the image
    image_path = f"{z}_grains_neck.png"
    image = cv2.imread(image_path)

    # Convert the image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Define color ranges for yellow and green
    # Adjust the ranges as necessary for your image
    yellow_lower = np.array([20, 100, 100])  # Lower bound for yellow in HSV
    yellow_upper = np.array([30, 255, 255])  # Upper bound for yellow in HSV

    green_lower = np.array([40, 50, 50])     # Lower bound for green in HSV
    green_upper = np.array([80, 255, 255])   # Upper bound for green in HSV

    # Create masks for yellow and green
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Step 3: Dilate the green mask to account for touching pixels
    kernel = np.ones((3, 3), np.uint8)  # Define a kernel size
    dilated_green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    # Find yellow pixels that are not touching green pixels
    isolated_yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(dilated_green_mask))

    # Step 4: Replace isolated yellow pixels with red
    # Create a red color array
    red_color = np.array([0, 0, 255], dtype=np.uint8)

    # Apply the mask to replace isolated yellow pixels with red
    image[isolated_yellow_mask > 0] = red_color

    # Step 5: Save the processed image
    output_path = f"{z}_grains_neck_1pix_yellow_neck.png"
    cv2.imwrite(output_path, image)

    print(f"Processed image saved to {output_path}")











    import cv2
    import numpy as np

    # Step 1: Load the image
    image_path = f"{z}_grains_neck_yellow_black.png"
    image = cv2.imread(image_path)

    # Convert the image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Define color ranges for yellow and green
    # Adjust the ranges as necessary for your image
    yellow_lower = np.array([20, 100, 100])  # Lower bound for yellow in HSV
    yellow_upper = np.array([30, 255, 255])  # Upper bound for yellow in HSV

    green_lower = np.array([40, 50, 50])     # Lower bound for green in HSV
    green_upper = np.array([80, 255, 255])   # Upper bound for green in HSV

    # Create masks for yellow and green
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Step 3: Dilate the green mask to account for touching pixels
    kernel = np.ones((3, 3), np.uint8)  # Define a kernel size
    dilated_green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    # Find yellow pixels that are not touching green pixels
    isolated_yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(dilated_green_mask))

    # Step 4: Replace isolated yellow pixels with red
    # Create a red color array
    red_color = np.array([0, 0, 255], dtype=np.uint8)

    # Apply the mask to replace isolated yellow pixels with red
    image[isolated_yellow_mask > 0] = red_color

    # Step 5: Save the processed image
    output_path = f"{z}_grains_neck_yellow_black_1pix_yellow_neck.png"
    cv2.imwrite(output_path, image)

    print(f"Processed image saved to {output_path}")


















    import cv2
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    # Load the image
    image = cv2.imread(f"{z}_grains_neck_yellow_black_1pix_yellow_neck.png")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV (for detecting grains)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Define the range for yellow color in HSV (for detecting necks)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Threshold the image to isolate green (grain) and yellow (neck) regions
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the green (grain) regions
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get scale information
    #croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
    #croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
    #conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel

    # Initialize variables
    valid_contours = 0
    contour_data = []

    # Kernel for morphological dilation
    kernel = np.ones((3, 3), np.uint8)

    # Process contours and store data
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < 1:  # Skip small contours
            continue

        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Create a mask for the current grain
        grain_mask = np.zeros(mask_green.shape, dtype=np.uint8)
        cv2.drawContours(grain_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Dilate the grain mask to find adjacent yellow pixels
        dilated_grain_mask = cv2.dilate(grain_mask, kernel, iterations=1)
        yellow_necks = cv2.bitwise_and(dilated_grain_mask, mask_yellow)

        # Count yellow pixels directly in contact with the grain
        neck_pixel_count = cv2.countNonZero(yellow_necks)

        # Calculate the total length of necks in µm
        total_neck_length = neck_pixel_count * conversion_factor

        # Calculate the perimeter of the grain in µm
        perimeter = cv2.arcLength(contour, True) * conversion_factor

        # Calculate contiguity
        contiguity = total_neck_length / perimeter if perimeter != 0 else 0

        # Update valid contours
        valid_contours += 1
        contour_data.append((contour, cX, cY, valid_contours, total_neck_length, perimeter, contiguity))

    # Prepare the output image for displaying the labels
    output_image = image.copy()

    # Draw contours and label numbers
    for contour, cX, cY, region_number, total_neck_length, perimeter, contiguity in contour_data:
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        text = str(region_number)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        textX = cX - text_size[0] // 2
        textY = cY + text_size[1] // 2
        cv2.putText(output_image, text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the labeled image
    cv2.imwrite(f"{z}_grain_neck_regions1.png", output_image)
    #cv2.imshow("Labeled Image", output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save output to Excel
    df = pd.DataFrame(contour_data, columns=['Contour', 'Centroid X', 'Centroid Y', 'Region', 'Total Neck Length (µm)', 'Perimeter (µm)', 'Contiguity'])
    df.drop(columns=['Contour'], inplace=True)  # Drop the contour column for cleaner Excel output
    df.to_excel(f"{z}_enclosed_grain_neck_contiguity_details.xlsx", index=False)

    # Final output for all grains
    if valid_contours == 0:
        print("No valid contours found.")
    else:
        print("\nSummary of Contiguities:")
        for data in contour_data:
            region, neck_length, perimeter, contiguity = data[3], data[4], data[5], data[6]
            print(f"Region {region}: Total Neck Length = {neck_length:.2f} µm, Perimeter = {perimeter:.2f} µm, Contiguity = {contiguity:.4f}")

            
      

    import pandas as pd
    import numpy as np
    from openpyxl import load_workbook

    # Load the Excel file
    file_name = f"{z}_enclosed_grain_neck_contiguity_details.xlsx"
    sheet_name = "Sheet1"  # Adjust if the sheet has a different name
    data = pd.read_excel(file_name, sheet_name=sheet_name)

    # Assuming the angles are in a column named 'Angles'
    contiguity = data['Contiguity']

    # Calculate the average and standard deviation
    average_contiguity11_ = np.mean(contiguity)
    std_dev11_ = np.std(contiguity)

    # Load the workbook and select the sheet
    wb = load_workbook(file_name)
    sheet = wb[sheet_name]

    # Write the results to the Excel file (e.g., in cells D1 and D2)
    sheet["H1"] = "Average Contiguity"
    sheet["I1"] = f"{average_contiguity11_:.2f} "
    sheet["H2"] = "Standard Deviation"
    sheet["I2"] = f"{std_dev11_:.2f} "

    # Save the updated workbook
    wb.save(file_name)



    ########extract yellow pixels



    import cv2
    import numpy as np

    # Step 1: Read the image
    image = cv2.imread(f"{z}_grains_neck.png")

    # Step 2: Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 3: Define the HSV range for yellow
    # Note: You may need to adjust these values based on your image
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Step 4: Create a mask for yellow pixels
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Step 5: Convert yellow pixels to white and background to black
    result_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    result_image[yellow_mask > 0] = [255, 255, 255]

    # Step 6: Convert the result to grayscale
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Step 7: Save or display the result
    #cv2.imshow('Result', gray_image)
    cv2.imwrite(f"{z}_grains_neck_ex.png", gray_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()






    #######thin img


    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage import img_as_ubyte

    # Load the image in grayscale mode
    image = cv2.imread(f"{z}_grains_neck_ex.png", cv2.IMREAD_GRAYSCALE)

    # Ensure the white boundary is represented as 255 and the background as 0
    # If your white boundary is not pure white (255), you may need to threshold it
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological erosion to thin the boundary
    # The kernel defines the neighborhood size for erosion, adjust if necessary
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=0)

    # Skeletonize the image to ensure the boundary is thinned
    skeleton = skeletonize(eroded_image // 255)  # Skeletonize expects a binary image (0, 1)
    skeleton_image = img_as_ubyte(skeleton)  # Convert back to 0-255 range

    # Dilate the skeleton to ensure uniform 2-pixel thickness boundary
    thinned_image = cv2.dilate(skeleton_image, kernel, iterations=0)

    # Save the output image

    # Display the result (optional)

    cv2.imwrite(f"{z}_grains_neck_ex_thin.png", thinned_image)
    #cv2.imshow("image",thinned_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()






    #######filling yellow pixels with green

    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_grains_neck.png")

    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for yellow color
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype=np.uint8)

    # Create a mask for yellow pixels
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Define the green color in BGR
    green_color = [0, 255, 0]

    # Replace yellow pixels with green
    image[yellow_mask > 0] = green_color

    # Save the modified image
    cv2.imwrite(f"{z}_grains_neck_yellow_to_green.png", image)

    # Display the original and modified images (optional)

    #cv2.imshow('Modified Image', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




    import cv2
    import numpy as np
    from skimage.morphology import disk, dilation
    from skimage import io, img_as_ubyte
    from skimage.color import rgb2gray
    from skimage.feature import canny

    # Read the original image
    original_image = cv2.imread(f"{z}_grains_neck_yellow_to_green.png")

    # Convert to grayscale
    gray_image = rgb2gray(original_image)

    # Detect edges using the Canny method
    edge_image = canny(gray_image)

    # Define a structuring element for dilation (disk of radius 0)
    selem = disk(0)

    # Dilate the edges
    dilated_edges = dilation(edge_image, selem)

    # Convert the dilated edge image to uint8 type for proper overlay
    dilated_edges_uint8 = img_as_ubyte(dilated_edges)

    # Overlay the dilated edges on the original image (in white)
    overlayed_image = original_image.copy()
    overlayed_image[dilated_edges_uint8 > 0] = [255, 255, 255]

    # Display the dilated edges


    # Save the result image
    cv2.imwrite(f"{z}_grains_neck_yellow_to_green_perimeter.png", overlayed_image)





    ####merge img contiguity

    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_grains_neck_yellow_to_green_perimeter.png")
    boundaryImg = cv2.imread(f"{z}_grains_neck_ex_thin.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [255, 255, 255]
    # Display the merged image


    cv2.imwrite(f"{z}_grains_neck_merged.png", outputImg)






    ###########contiguity


    #num_rows=int(input("Enter number of lines for enclosed grain contiguity : "))
    #import cv2
    #import numpy as np
    #import pandas as pd  # For saving data to Excel
    #import matplotlib.pyplot as plt

    #def draw_line_and_mark_intersections(image, row_index):
        # Define the colors
        #tungsten_color = np.array([0, 255, 0])  # green
        #boundary_color = np.array([255, 255, 255])  # white
        #matrix_color = np.array([255, 0, 0])  # red
        
        # Extract the row from the image
        #row = image[row_index, :, :]
        
        # Initialize lists to store intersection points
        #intersections = []
        
        # Process the row to find intersections with white pixels
        #for i in range(row.shape[0]):
            #pixel = row[i]
            #if np.array_equal(pixel, boundary_color):
                #intersections.append((row_index, i))  # Mark this point for intersection
        
        # Initialize interaction counters
        #tungsten_tungsten_interactions = 0
        ##tungsten_matrix_interactions = 0
        #matrix_tungsten_interactions = 0
        
        # Process intersections to count interactions
        #for point in intersections:
            #y, x = point
            
            # Check left and right pixels
            #left_pixel = row[x - 1] if x - 1 >= 0 else None
            #right_pixel = row[x + 1] if x + 1 < row.shape[0] else None
            
            #if left_pixel is not None and right_pixel is not None:
                #if np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, tungsten_color):
                    #tungsten_tungsten_interactions += 1
                #elif np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, matrix_color):
                    #tungsten_matrix_interactions += 1
                #elif np.array_equal(left_pixel, matrix_color) and np.array_equal(right_pixel, tungsten_color):
                    #matrix_tungsten_interactions += 1

        # Check for cases with consecutive boundary pixels
        #for i in range(len(intersections) - 1):
            #curr_point = intersections[i]
            #next_point = intersections[i + 1]

            # Left pixel of current point and right pixel of next point
            #left_pixel = row[curr_point[1] - 1] if curr_point[1] - 1 >= 0 else None
            #right_pixel = row[next_point[1] + 1] if next_point[1] + 1 < row.shape[0] else None
            
            #if left_pixel is not None and right_pixel is not None:
                #if np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, matrix_color):
                    #tungsten_matrix_interactions += 1
                #elif np.array_equal(left_pixel, matrix_color) and np.array_equal(right_pixel, tungsten_color):
                    #matrix_tungsten_interactions += 1

        # Calculate contiguity
        #denominator = (2 * tungsten_tungsten_interactions + tungsten_matrix_interactions + matrix_tungsten_interactions)
        #if denominator != 0:
        #    contiguity = (2 * tungsten_tungsten_interactions) / denominator
        #else:
            #contiguity = 0
        
        #return contiguity

    #def calculate_average_contiguity(image_path, num_rows, output_excel=f"{z}_enclosed_grain_linear_contiguity_details.xlsx"):
        # Load the image
        #image = cv2.imread(image_path)
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Determine the factor
        #factor = image_rgb.shape[0] / num_rows
        
        # Create a DataFrame to store results
        ##contiguity_data = []

        # Calculate contiguity for each row and take the average
        #total_contiguity = 0
        #valid_rows = 0  # Count rows with contiguity > 0.1
        #for i in range(num_rows):
            ##row_index = int(i * factor)
           # contiguity = draw_line_and_mark_intersections(image_rgb, row_index)
            #i#f contiguity >= 0:  # Filter for contiguity > 0.1
               # total_contiguity += contiguity
               ## valid_rows += 1
                #print(f"Row {row_index}: Contiguity = {contiguity}")
                ## Append row index and contiguity to the data list
                #contiguity_data.append({"Row Index": row_index, "Contiguity": contiguity})
        
        # Calculate average contiguity
        #if valid_rows > 0:
            #average_contiguity = total_contiguity / valid_rows
       # else:
           # average_contiguity = 0  # Handle case where no rows meet the condition
        
        # Save the data to Excel
       ## df = pd.DataFrame(contiguity_data)
        #df.to_excel(output_excel, index=False)
        
        # Calculate standard deviation
        #std_dev1= np.std(df['Contiguity'])
        
        #return average_contiguity, std_dev1


    # Example usage
    ##image_path = f"{z}_grains_neck_merged.png"  # Change this to the path of your image
    #average_contiguity, std_dev1 = calculate_average_contiguity(image_path, num_rows, output_excel=f"{z}_enclosed_grain_linear_contiguity_details.xlsx")



    #import pandas as pd
    ##import numpy as np
    #from openpyxl import load_workbook

    # Load the Excel file
    #file_name = f"{z}_enclosed_grain_linear_contiguity_details.xlsx"
    #sheet_name = "Sheet1"  # Adjust if the sheet has a different name
    #data = pd.read_excel(file_name, sheet_name=sheet_name)

    # Assuming the angles are in a column named 'Angles'
   # contiguity = data['Contiguity']

    # Calculate the average and standard deviation
    #average_contiguity11 = np.mean(contiguity)
    ##std_dev11 = np.std(contiguity)

    # Load the workbook and select the sheet
    #wb = load_workbook(file_name)
    #sheet = wb[sheet_name]

    # Write the results to the Excel file (e.g., in cells D1 and D2)
    #sheet["D1"] = "Average Contiguity"
    ##sheet["E1"] = f"{average_contiguity11:.2f} "
    #sheet["D2"] = "Standard Deviation"
    #sheet["E2"] = f"{std_dev11:.2f} "

    # Save the updated workbook
    #wb.save(file_name)









    num_rows=int(input("Enter number of lines to be used for contiguity measurement : "))

    import cv2
    import numpy as np
    import pandas as pd  # For saving data to Excel
    import matplotlib.pyplot as plt

    def draw_line_and_mark_intersections(image, row_index):
        # Define the colors
        tungsten_color = np.array([0, 255, 0])  # green
        boundary_color = np.array([255, 255, 255])  # white
        matrix_color = np.array([255, 0, 0])  # red
        
        # Extract the row from the image
        row = image[row_index, :, :]
        
        # Initialize lists to store intersection points
        intersections = []
        
        # Process the row to find intersections with white pixels
        for i in range(row.shape[0]):
            pixel = row[i]
            if np.array_equal(pixel, boundary_color):
                intersections.append((row_index, i))  # Mark this point for intersection
        
        # Initialize interaction counters
        tungsten_tungsten_interactions = 0
        tungsten_matrix_interactions = 0
        matrix_tungsten_interactions = 0
        
        # Process intersections to count interactions
        for point in intersections:
            y, x = point
            
            # Check left and right pixels
            left_pixel = row[x - 1] if x - 1 >= 0 else None
            right_pixel = row[x + 1] if x + 1 < row.shape[0] else None
            
            if left_pixel is not None and right_pixel is not None:
                if np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, tungsten_color):
                    tungsten_tungsten_interactions += 1
                elif np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, matrix_color):
                    tungsten_matrix_interactions += 1
                elif np.array_equal(left_pixel, matrix_color) and np.array_equal(right_pixel, tungsten_color):
                    matrix_tungsten_interactions += 1

        # Check for cases with consecutive boundary pixels
        for i in range(len(intersections) - 1):
            curr_point = intersections[i]
            next_point = intersections[i + 1]

            # Left pixel of current point and right pixel of next point
            left_pixel = row[curr_point[1] - 1] if curr_point[1] - 1 >= 0 else None
            right_pixel = row[next_point[1] + 1] if next_point[1] + 1 < row.shape[0] else None
            
            if left_pixel is not None and right_pixel is not None:
                if np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, matrix_color):
                    tungsten_matrix_interactions += 1
                elif np.array_equal(left_pixel, matrix_color) and np.array_equal(right_pixel, tungsten_color):
                    matrix_tungsten_interactions += 1

        # Calculate contiguity
        denominator = (2 * tungsten_tungsten_interactions + tungsten_matrix_interactions + matrix_tungsten_interactions)
        if denominator != 0:
            contiguity = (2 * tungsten_tungsten_interactions) / denominator
        else:
            contiguity = 0
        
        return contiguity

    def calculate_average_contiguity(image_path, num_rows, output_excel=f"{z}_contiguity_results.xlsx"):
        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Determine the factor
        factor = image_rgb.shape[0] / num_rows
        
        # Create a DataFrame to store results
        contiguity_data = []

        # Calculate contiguity for each row and take the average
        total_contiguity = 0
        valid_rows = 0  # Count rows with contiguity > 0.1
        for i in range(num_rows):
            row_index = int(i * factor)
            contiguity = draw_line_and_mark_intersections(image_rgb, row_index)
            if contiguity >= 0:  # Filter for contiguity > 0.1
                total_contiguity += contiguity
                valid_rows += 1
                print(f"Row {row_index}: Contiguity = {contiguity}")
                # Append row index and contiguity to the data list
                contiguity_data.append({"Row Index": row_index, "Contiguity": contiguity})
        
        # Calculate average contiguity
        if valid_rows > 0:
            average_contiguity = total_contiguity / valid_rows
        else:
            average_contiguity = 0  # Handle case where no rows meet the condition
        
        # Save the data to Excel
        df = pd.DataFrame(contiguity_data)
        df.to_excel(output_excel, index=False)
        
        # Calculate standard deviation
        std_dev1= np.std(df['Contiguity'])
        
        return average_contiguity, std_dev1


    # Example usage
    image_path = f"{z}_merged_thin_img.png"  # Change this to the path of your image
    average_contiguity, std_dev1 = calculate_average_contiguity(image_path, num_rows, output_excel=f"{z}_contiguity_results.xlsx")



    import pandas as pd
    import numpy as np
    from openpyxl import load_workbook

    # Load the Excel file
    file_name = f"{z}_contiguity_results.xlsx"
    sheet_name = "Sheet1"  # Adjust if the sheet has a different name
    data = pd.read_excel(file_name, sheet_name=sheet_name)

    # Assuming the angles are in a column named 'Angles'
    contiguity = data['Contiguity']

    # Calculate the average and standard deviation
    average_contiguity11 = np.mean(contiguity)
    std_dev11 = np.std(contiguity)

    # Load the workbook and select the sheet
    wb = load_workbook(file_name)
    sheet = wb[sheet_name]

    # Write the results to the Excel file (e.g., in cells D1 and D2)
    sheet["D1"] = "Average Contiguity"
    sheet["E1"] = f"{average_contiguity11:.2f} "
    sheet["D2"] = "Standard Deviation"
    sheet["E2"] = f"{std_dev11:.2f} "

    # Save the updated workbook
    wb.save(file_name)
    
    
    
    image = cv2.imread(f"{z}.png")

    # Convert the image to grayscale for thresholding
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the thresholded image using matplotlib
    plt.imshow(thresh, cmap='gray')
    plt.show()

    #cv2.imshow("threshold_image",thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save the thresholded image
    plt.imsave(f"{z}_thresh_precipitate.png", thresh, cmap='gray')


    img=cv2.imread(f"{z}_thresh_precipitate.png",0)
    color1=img==0
    color2=img==255
    allsegment=np.zeros((img.shape[0],img.shape[1],3))
    allsegment[color1]=(0,0,0)
    allsegment[color2]=(0,1,0)
    plt.imsave(f"{z}_thresh_precipitate_green.png",allsegment)




    import cv2
    import numpy as np

    def filter_green_clusters(image_path, min_pixels, max_pixels):
        # Step 1: Load the image in color
        image = cv2.imread(image_path)
        
        # Step 2: Create a mask for green pixels (threshold for green)
        # Define the lower and upper bounds for the green color in BGR format
        lower_green = np.array([0, 100, 0])  # Lower bound for green (BGR)
        upper_green = np.array([100, 255, 100])  # Upper bound for green (BGR)

        # Threshold the image to get only green pixels
        green_mask = cv2.inRange(image, lower_green, upper_green)

        # Step 3: Find connected components for green pixels
        num_labels, labels_im = cv2.connectedComponents(green_mask)

        # Create an output image, initially all black
        output_image = np.zeros_like(green_mask)

        # Step 4: Filter the connected components based on the pixel count range
        for label in range(1, num_labels):  # Start from 1 because 0 is the background
            # Get all the pixels that belong to this component
            component = (labels_im == label).astype(np.uint8)

            # Count the number of green pixels in this component
            pixel_size = np.sum(component)

            # Step 5: If the component size is within the specified range, retain it
            if min_pixels <= pixel_size <= max_pixels:
                output_image[component == 1] = 255  # Set pixels of this component to white

        # Step 6: Save and display the output image
        cv2.imwrite(f"{z}_thresh_precipitate_particles.png", output_image)
        #cv2.imshow('Filtered Green Clusters', output_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    # Usage example
    image_path = f"{z}_thresh_precipitate_green.png"  # Path to the input image
    filter_green_clusters(image_path, 0, max_precipitate_size_pixel)






    img=cv2.imread(f"{z}_thresh_precipitate_particles.png",0)
    seg111=img==0
    seg222=img==255
    allsegment=np.zeros((img.shape[0],img.shape[1],3))
    allsegment[seg111]=(1,0,0)
    allsegment[seg222]=(0,1,0)

    if (np.sum(seg222==1)*100)/(np.sum(seg222==1)+np.sum(seg222==0))==0:
        import os

        # Specify the path to the file
        file_path = f"{z}_thresh_precipitate_particles.png"

        # Check if the file exists before attempting to delete
        if os.path.exists(file_path):
           os.remove(file_path)    
        print("There are no Tungsten precipitate in the Micrograph")
    else:
        print("The are Tungsten precipitates")
        
        img=cv2.imread(f"{z}_thresh_precipitate_particles.png",0)
        seg_1_1=img==0
        seg_2_2=img==255
        allsegment=np.zeros((img.shape[0],img.shape[1],3))
        allsegment[seg_1_1]=(1,0,0)
        allsegment[seg_2_2]=(0,1,0)
        plt.imsave(f"{z}_thresh_precipitate_particles_seg.png",allsegment)





        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        # Load the image
        image = cv2.imread(f"{z}_thresh_precipitate_particles_seg.png")

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])

        # Threshold the image to isolate green regions
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours of the green regions
        contours_, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image for drawing
        output_image = image.copy()

        # Get scale information
        #croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
        #croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
        #conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel

        # Initialize variables
        valid_contours_ = 0
        total_major_diameter_ = 0
        contour_data_ = []
        major_diameters_ = []

        # Process contours
        for i, contour_ in enumerate(contours_, 1):
            if cv2.contourArea(contour_) < 0:  # Skip small contours
                continue

            moments_ = cv2.moments(contour_)
            if moments_['m00'] != 0:
                cX = int(moments_['m10'] / moments_['m00'])
                cY = int(moments_['m01'] / moments_['m00'])

                # Bounding box and major diameter
                x, y, w, h = cv2.boundingRect(contour_)
                major_diameter_ = max(w, h) * conversion_factor  # µm
                area_um2_ = cv2.contourArea(contour_) * conversion_factor ** 2
                perimeter_um_ = cv2.arcLength(contour_, True) * conversion_factor
                particle_edge_um_ = area_um2_ ** 0.5
                circularity_um_ = 4 * 3.14 * area_um2_ / (perimeter_um_ ** 2)

                # Update variables
                total_major_diameter_ += major_diameter_
                major_diameters_.append(major_diameter_)
                contour_data_.append((contour_, cX, cY, major_diameter_))
                valid_contours_ += 1

                # Output details
                print(f'Region {valid_contours_}:')
                print(f'  Area: {area_um2_:.2f} µm²')
                print(f'  Perimeter: {perimeter_um_:.2f} µm')
                print(f'  Major Diameter: {major_diameter_:.2f} µm')
                print(f'  Particle Edge: {particle_edge_um_:.2f} µm')
                print(f'  Circularity: {circularity_um_:.2f} µm')
                print()

        # Final calculations
        if valid_contours_ > 0:
            average_major_diameter_ = total_major_diameter_ / valid_contours_
            min_diameter_ = min(major_diameters_)
            max_diameter_ = max(major_diameters_)
            # Calculate the standard deviation using numpy
            std_dev_ = np.std(major_diameters_)
            print(f"\nAverage Major Diameter: {average_major_diameter_:.2f} µm")
            print(f"Minimum Major Diameter: {min(major_diameters_):.2f} µm")
            print(f"Maximum Major Diameter: {max(major_diameters_):.2f} µm")
            print(f"Standard Deviation: {np.std(major_diameters_):.2f} µm")

        # Draw contours and label numbers
        for i, (contour_, cX, cY, major_diameter_) in enumerate(contour_data_, 1):
            cv2.drawContours(output_image, [contour_], -1, (0, 255, 0), 2)
            text_ = str(i)
            text_size_ = cv2.getTextSize(text_, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            textX_ = cX - text_size_[0] // 2
            textY_ = cY + text_size_[1] // 2
            cv2.putText(output_image, text_, (textX_, textY_), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save and display results
        cv2.imwrite(f"{z}_thresh_precipitate_particles_regions.png", output_image)
        #cv2.imshow("Labeled Image", output_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Particle size distribution
        if major_diameters_:
            major_diameters_ = np.array(major_diameters_)
            hist, bins = np.histogram(major_diameters_, bins='auto')
            bin_centers_ = 0.5 * (bins[1:] + bins[:-1])
            mu, std = norm.fit(major_diameters_)

            plt.bar(bin_centers_, hist, width=(bins[1] - bins[0]), alpha=0.6, color='purple', edgecolor='black', linewidth=1.0)
            x = np.linspace(bin_centers_[0], bin_centers_[-1], 100)
            p = norm.pdf(x, mu, std) * len(major_diameters_) * (bins[1] - bins[0])
            plt.plot(x, p, 'b', linewidth=2)
            plt.title('W Precipitates Size Distribution')
            plt.xlabel('W Precipitates Size (µm)')
            plt.ylabel('Count')
            plt.xlim([0, max(major_diameters_) + 10])
            plt.ylim([0, max(hist) + 5])
            plt.savefig(f"{z}_thresh_precipitate_particles_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            
            
            
            
        import pandas as pd

        # Initialize a list to store the details of each valid region
        region_details = []
        valid_contours = 0  # Counter for valid regions

        # Process each contour and store the details
        for contour_ in contours_:
            if cv2.contourArea(contour_) < 0:  # Ignore small contours
                continue

            moments = cv2.moments(contour_)
            if moments['m00'] != 0:
                x, y, w, h = cv2.boundingRect(contour_)
                major_diameter_ = max(w, h) * conversion_factor  # Major diameter in µm
                area_um2_ = cv2.contourArea(contour_) * conversion_factor ** 2
                perimeter_um_ = cv2.arcLength(contour_, True) * conversion_factor
                particle_edge_um_ = area_um2_ ** 0.5  # Corrected to compute the square root of area_um2
                circularity_um_ = 4 * 3.14 * area_um2_ / (perimeter_um_ ** 2)  # Corrected to square the perimeter_um variable

                # Increment valid region counter
                valid_contours += 1

                # Append details to the list
                region_details.append({
                    'Region': valid_contours,  # Use valid_contours as region index
                    'Area (µm²)': round(area_um2_, 2),
                    'Perimeter (µm)': round(perimeter_um_, 2),
                    'Major Diameter (µm)': round(major_diameter_, 2),
                    'Particle Edge (µm)': round(particle_edge_um_, 2),
                    'Circularity (µm)': round(circularity_um_, 2)
                })

        # Ensure all valid regions match the image labels
        if len(region_details) != valid_contours:
            raise ValueError("Mismatch between the number of labeled regions and saved region details.")

        # Create a pandas DataFrame from the region details
        df = pd.DataFrame(region_details)

        # Save the DataFrame to an Excel file
        output_file = f"{z}_precipitates_region_details.xlsx"
        df.to_excel(output_file, index=False)

        print(f"Region details successfully saved to {output_file}")




        import pandas as pd
        import numpy as np
        from openpyxl import load_workbook

        # Load the Excel file
        file_name = f"{z}_precipitates_region_details.xlsx"
        sheet_name = "Sheet1"  # Adjust if the sheet has a different name
        data = pd.read_excel(file_name, sheet_name=sheet_name)

        # Assuming the angles are in a column named 'Angles'
        grain_data = data['Major Diameter (µm)']

        # Calculate the average and standard deviation
        average_grain_size22_ = np.mean(grain_data)
        std_dev22_= np.std(grain_data)

        # Load the workbook and select the sheet
        wb = load_workbook(file_name)
        sheet = wb[sheet_name]

        # Write the results to the Excel file (e.g., in cells D1 and D2)
        sheet["H1"] = "Average Precipitate Size (µm)"
        sheet["I1"] = f"{average_grain_size22_:.2f} "
        sheet["H2"] = "Standard Deviation (µm)"
        sheet["I2"] = f"{std_dev22_:.2f} "
        sheet["H3"] = "Min diameter (µm)"
        sheet["I3"] = f"{min_diameter_:.2f} "
        sheet["H4"] = "Max diameter (µm)"
        sheet["I4"] = f"{max_diameter_:.2f} "

        # Save the updated workbook
        wb.save(file_name)


    
    
    
    
    
    ############################################################DI-HYDRAL ANGLE
    
    
    ############################Finding boundary perimeter of a segmented image or threshold image
    import cv2
    import numpy as np
    from skimage.morphology import disk, dilation
    from skimage import io, img_as_ubyte
    from skimage.color import rgb2gray
    from skimage.feature import canny
    
    # Read the original image
    original_image = cv2.imread(f"{z}_thresh_seg.png")
    
    # Convert to grayscale
    gray_image = rgb2gray(original_image)
    
    # Detect edges using the Canny method
    edge_image = canny(gray_image)
    
    # Define a structuring element for dilation (disk of radius 0)
    selem = disk(0)
    
    # Dilate the edges
    dilated_edges = dilation(edge_image, selem)
    
    # Convert the dilated edge image to uint8 type for proper overlay
    dilated_edges_uint8 = img_as_ubyte(dilated_edges)
    
    # Overlay the dilated edges on the original image (in white)
    overlayed_image = original_image.copy()
    overlayed_image[dilated_edges_uint8 > 0] = [255, 255, 255]
    
    # Display the dilated edges
    
    
    # Save the result image
    cv2.imwrite(f"{z}_perimeter.png", overlayed_image)
    
    
    
    
    
    
    
    ##################################### 1st condition
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter.png")
    
    # Define color ranges
    white_pixel = [255, 255, 255]  # White boundary pixel
    green_pixel = [0, 255, 0]  # Green pixel
    blue_pixel = [255, 0, 0]  # Blue pixel (BGR format)
    
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Check conditions and fill the white boundary pixel with blue
    def is_green(pixel):
        return np.array_equal(pixel, green_pixel)
    
    def is_white(pixel):
        return np.array_equal(pixel, white_pixel)
    
    # Loop through each pixel to find white boundary pixels and check conditions
    for y in range(1, height-1):  # Avoiding the edges
        for x in range(1, width-1):  # Avoiding the edges
            if is_white(image[y, x]):
                # Check horizontal direction (left and right)
                left_green = all(is_green(image[y, x-i]) for i in range(3, 8) if x-i >= 0) and is_white(image[y, x-1])
                right_green = all(is_green(image[y, x+i]) for i in range(3, 8) if x+i < width) and is_white(image[y, x+1])
                
                # Check vertical direction (up and down)
                up_green = all(is_green(image[y-i, x]) for i in range(3, 8) if y-i >= 0) and is_white(image[y-1, x])
                down_green = all(is_green(image[y+i, x]) for i in range(3, 8) if y+i < height) and is_white(image[y+1, x])
                
                # If either condition is met, fill the white pixel with blue
                if (left_green and right_green) or (up_green and down_green):
                    image[y, x] = blue_pixel
    
    # Save the modified image
    cv2.imwrite(f"{z}_perimeter_1st.png", image)
    
    
    
    
    
    
    
    ###################### 2nd condition
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter.png")
    
    # Convert to RGB (OpenCV loads images in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the pixel values for colors
    WHITE = [255, 255, 255]  # White boundary pixels
    GREEN = [0, 255, 0]      # Green grain pixels
    BLUE = [0, 0, 255]       # Blue for filling the selected boundary pixels
    
    # Function to check the conditions
    def check_conditions(image, x, y):
        # Ensure we are not on the edge of the image
        if x - 7 >= 0 and y - 7 >= 0 and x + 7 < image.shape[1] and y + 7 < image.shape[0]:
            # Extract the pixel neighbors #3 best white pixels
            right_4 = image[y, x+1:x+4]
            left_7 = image[y, x-7:x]
            down_4 = image[y+1:y+4, x]
            up_7 = image[y-7:y, x]
            
            right_7 = image[y, x+1:x+8]
            left_4 = image[y, x-3:x]
            down_7 = image[y+1:y+8, x]
            up_4 = image[y-3:y, x]
            
            # Check the four conditions
            if (np.all(right_4 == WHITE) and np.all(left_7 == GREEN) and np.all(down_4 == WHITE) and np.all(up_7 == GREEN)):
                return True
            if (np.all(right_7 == GREEN) and np.all(left_4 == WHITE) and np.all(down_4 == WHITE) and np.all(up_7 == GREEN)):
                return True
            if (np.all(right_7 == GREEN) and np.all(left_4 == WHITE) and np.all(down_7 == GREEN) and np.all(up_4 == WHITE)):
                return True
            if (np.all(right_4 == WHITE) and np.all(left_7 == GREEN) and np.all(down_7 == GREEN) and np.all(up_4 == WHITE)):
                return True
        return False
    
    # Loop over the image to find individual white pixels
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Check if the current pixel is white
            if np.all(image_rgb[y, x] == WHITE):
                # Check if the conditions are met
                if check_conditions(image_rgb, x, y):
                    # Fill with blue if conditions are met
                    image_rgb[y, x] = BLUE
    
    # Convert the image back to BGR and save
    result_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{z}_perimeter_2nd.png", result_image)
    
    
    
    
    
    
    
    
    ###Extraction of blue and white
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load the image
    image_path = f"{z}_perimeter_1st.png"
    image = cv2.imread(image_path)
    
    # Convert the image to HSV (Hue, Saturation, Value) for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for white and blue pixels in HSV
    # White color range
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 55, 255], dtype=np.uint8)
    
    # Blue color range
    lower_blue = np.array([100, 150, 0], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    
    # Create masks for white and blue colors
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Combine the two masks
    combined_mask = cv2.bitwise_or(white_mask, blue_mask)
    
    # Extract only the white and blue pixels, setting the rest to black
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    
    # Save the result image
    cv2.imwrite(f"{z}_perimeter_1st_ex.png", result)
    
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load the image
    image_path = f"{z}_perimeter_2nd.png"
    image = cv2.imread(image_path)
    
    # Convert the image to HSV (Hue, Saturation, Value) for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for white and blue pixels in HSV
    # White color range
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 55, 255], dtype=np.uint8)
    
    # Blue color range
    lower_blue = np.array([100, 150, 0], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    
    # Create masks for white and blue colors
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Combine the two masks
    combined_mask = cv2.bitwise_or(white_mask, blue_mask)
    
    # Extract only the white and blue pixels, setting the rest to black
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    
    # Save the result image
    cv2.imwrite(f"{z}_perimeter_2nd_ex.png", result)
    
    
    
    
    ################## adjusting blue pixels
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter_1st_ex.png")
    
    # Threshold the image to isolate white grain boundary and blue pixels
    # Blue pixel range (B, G, R): (255, 0, 0)
    blue_pixels = np.where((image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))
    
    # Create a copy of the image to modify
    output_image = image.copy()
    
    # Define function to calculate Euclidean distance between two points
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    # Process each blue pixel
    for i in range(len(blue_pixels[0])):
        x1, y1 = blue_pixels[0][i], blue_pixels[1][i]
        
        for j in range(i + 1, len(blue_pixels[0])):
            x2, y2 = blue_pixels[0][j], blue_pixels[1][j]
            
            # Check if the two blue pixels are between 2-5 pixels apart
            distance = euclidean_distance((x1, y1), (x2, y2))
            if 1 <= distance <= 5:
                # Calculate the midpoint
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Remove the original blue pixels
                output_image[x1, y1] = [255, 255, 255]  # Set to white (original boundary color)
                output_image[x2, y2] = [255, 255, 255]  # Set to white (original boundary color)
                
                # Place the resultant blue pixel at the midpoint
                output_image[mid_x, mid_y] = [255, 0, 0]  # Set to blue
    
    # Save the output image
    cv2.imwrite(f"{z}_perimeter_1st_ex_ad.png", output_image)
    
    
    
    
    
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter_2nd_ex.png")
    
    # Threshold the image to isolate white grain boundary and blue pixels
    # Blue pixel range (B, G, R): (255, 0, 0)
    blue_pixels = np.where((image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))
    
    # Create a copy of the image to modify
    output_image = image.copy()
    
    # Define function to calculate Euclidean distance between two points
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    # Process each blue pixel
    for i in range(len(blue_pixels[0])):
        x1, y1 = blue_pixels[0][i], blue_pixels[1][i]
        
        for j in range(i + 1, len(blue_pixels[0])):
            x2, y2 = blue_pixels[0][j], blue_pixels[1][j]
            
            # Check if the two blue pixels are between 2-5 pixels apart
            distance = euclidean_distance((x1, y1), (x2, y2))
            if 1 <= distance <= 5:
                # Calculate the midpoint
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Remove the original blue pixels
                output_image[x1, y1] = [255, 255, 255]  # Set to white (original boundary color)
                output_image[x2, y2] = [255, 255, 255]  # Set to white (original boundary color)
                
                # Place the resultant blue pixel at the midpoint
                output_image[mid_x, mid_y] = [255, 0, 0]  # Set to blue
    
    # Save the output image
    cv2.imwrite(f"{z}_perimeter_2nd_ex_ad.png", output_image)
    
    
    
    
    
    #############Extracting only blue 2nd condition
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter_2nd_ex_ad.png")
    
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for the blue color in HSV
    lower_blue = np.array([100, 150, 0])  # Lower bound for blue
    upper_blue = np.array([140, 255, 255])  # Upper bound for blue
    
    # Create a mask to extract blue regions
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Create a black background
    output_image = np.zeros_like(image)
    
    # Extract only the blue parts of the image and set the rest to black
    output_image[mask > 0] = image[mask > 0]
    
    # Save the result
    cv2.imwrite(f"{z}_perimeter_2nd_ex_ad_ex.png", output_image)
    
    
    
    
    
    ############ combined extractions
    
    import cv2
    import numpy as np
    
    # Load the two images
    image1 = cv2.imread(f"{z}_perimeter_1st_ex_ad.png")  # First image
    image2 = cv2.imread(f"{z}_perimeter_2nd_ex_ad_ex.png")  # Second image
    
    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        print("The images must have the same dimensions.")
        exit()
    
    # Define the blue color in BGR format
    blue = [255, 0, 0]  # Blue color in OpenCV (BGR format)
    
    # Create a mask to find the blue pixels in the second image
    blue_mask = (image2 == blue).all(axis=-1)
    
    # Add the blue pixels from the second image to the first image
    image1[blue_mask] = blue
    
    # Save the result
    cv2.imwrite(f"{z}_perimeter_1st_2nd.png", image1)
    
    
    
    
    ################## adjusting blue pixels
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter_1st_2nd.png")
    
    # Threshold the image to isolate white grain boundary and blue pixels
    # Blue pixel range (B, G, R): (255, 0, 0)
    blue_pixels = np.where((image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))
    
    # Create a copy of the image to modify
    output_image = image.copy()
    
    # Define function to calculate Euclidean distance between two points
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    # Process each blue pixel
    for i in range(len(blue_pixels[0])):
        x1, y1 = blue_pixels[0][i], blue_pixels[1][i]
        
        for j in range(i + 1, len(blue_pixels[0])):
            x2, y2 = blue_pixels[0][j], blue_pixels[1][j]
            
            # Check if the two blue pixels are between 2-5 pixels apart
            distance = euclidean_distance((x1, y1), (x2, y2))
            if 1 <= distance <= 5:
                # Calculate the midpoint
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Remove the original blue pixels
                output_image[x1, y1] = [255, 255, 255]  # Set to white (original boundary color)
                output_image[x2, y2] = [255, 255, 255]  # Set to white (original boundary color)
                
                # Place the resultant blue pixel at the midpoint
                output_image[mid_x, mid_y] = [255, 0, 0]  # Set to blue
    
    # Save the output image
    cv2.imwrite(f"{z}_perimeter_1st_2nd_ad.png", output_image)
    
    cv2.imshow("adjusted_image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    condition=int(input("Do you wish to edit the identified vertex points in the image then enter 0 otherwise enter 1 : "))
    if condition==0:
        
        
        
        import cv2
        import numpy as np

        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad.png")
        original_image = image.copy()  # Keep a copy of the original image for reset

        # Zoom parameters
        zoom_factor = 1.0
        zoom_step = 0.1
        min_zoom = 0.5
        max_zoom = 3.0

        # Flags and settings
        drawing = False  # True if the mouse is pressed
        erasing = False  # Toggle between drawing and erasing
        brush_size = 2  # Brush size for drawing or erasing
        color = (255, 255, 255)  # Default color (white)
        undo_stack = [image.copy()]  # Stack to store undo history

        # Mouse callback function
        def draw(event, x, y, flags, param):
            global drawing, erasing, image, zoom_factor
            
            # Convert zoomed coordinates back to the original coordinates
            zoom_x = int(x / zoom_factor)
            zoom_y = int(y / zoom_factor)
            
            # Start drawing/erasing when left mouse button is pressed
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                undo_stack.append(image.copy())  # Save the current image state for undo
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
            
            # Draw or erase as the mouse moves
            if event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    if not erasing:  # Drawing mode
                        cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                    else:  # Erasing mode
                        cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

        # Function to apply zoom
        def apply_zoom(image, zoom_factor):
            h, w = image.shape[:2]
            zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
            return zoomed_image

        # Trackbar callback function for brush size
        def set_brush_size(val):
            global brush_size
            brush_size = val

        # Trackbar callback function for color selection
        def set_color(val):
            global color
            if val == 0:
                color = (255, 0, 0)  # Blue
            elif val == 1:
                color = (255, 255, 255)  # White

        # Create a window and set mouse callback
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', draw)

        # Create trackbars for brush size and color selection
        cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
        cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

        while True:
            # Apply zoom to the image before displaying
            zoomed_image = apply_zoom(image, zoom_factor)
            cv2.imshow('Image', zoomed_image)
            
            # Wait for keypress
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('e'):  # Press 'e' to toggle erasing mode
                erasing = not erasing
                print('Erasing mode:', erasing)
            
            elif key == ord('r'):  # Press 'r' to reset the image
                image = original_image.copy()
                undo_stack.clear()
                undo_stack.append(image.copy())
                print('Image reset')
            
            elif key == ord('q'):  # Press 'q' to quit and save the image
                # Save the image before quitting
                cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit.png", image)
                break
            
            elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
                zoom_factor = min(zoom_factor + zoom_step, max_zoom)
                print('Zooming in:', zoom_factor)
            
            elif key == ord('-'):  # Press '-' to zoom out
                zoom_factor = max(zoom_factor - zoom_step, min_zoom)
                print('Zooming out:', zoom_factor)
            
            elif key == ord('u'):  # Press 'u' for undo
                if len(undo_stack) > 1:
                    undo_stack.pop()  # Remove the current state
                    image = undo_stack[-1].copy()  # Restore the previous state
                    print('Undo last action')

        # Close the window
        cv2.destroyAllWindows()
        
        
        #######important RED FILLING
        import cv2
        import numpy as np

        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit.png") 

        # Define color ranges (BGR format for OpenCV)
        blue_lower = np.array([100, 0, 0])    # Lower bound for blue
        blue_upper = np.array([255, 50, 50])  # Upper bound for blue
        white_lower = np.array([200, 200, 200])  # Lower bound for white
        white_upper = np.array([255, 255, 255])  # Upper bound for white

        # Find blue pixels
        blue_mask = cv2.inRange(image, blue_lower, blue_upper)

        # Find white pixels
        white_mask = cv2.inRange(image, white_lower, white_upper)

        # Temporary mask to mark red areas
        red_mask = np.zeros_like(blue_mask)

        # Get coordinates of blue pixels
        blue_pixels = np.argwhere(blue_mask > 0)

        # Set a radius of 10 pixels around each blue pixel and find white pixels nearby
        for blue_pixel in blue_pixels:
            y, x = blue_pixel
            # Define the region of interest (ROI) around each blue pixel
            y_min = max(0, y - 6) 
            y_max = min(image.shape[0], y + 6) 
            x_min = max(0, x - 6) 
            x_max = min(image.shape[1], x + 6) 
            
            # Extract the ROI from the white mask
            roi = white_mask[y_min:y_max, x_min:x_max]
            
            # Find coordinates of white pixels in the ROI
            white_pixels_in_roi = np.argwhere(roi > 0)
            
            # Mark white pixels in red_mask with a value of 255 (to signify red)
            for wp in white_pixels_in_roi:
                roi_y, roi_x = wp
                red_mask[y_min + roi_y, x_min + roi_x] += 1

        # Now, fill red in the original image only in locations where red_mask == 1
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if red_mask[y, x] == 1:  # If no overlap, mark as red
                    image[y, x] = [0, 0, 255]  # Red in BGR

        # Save or display the modified image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Red in BGR
        
        # Save the modified image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill.png", image)
        condition=int(input("If you wish to make changes Enter ‘0’ otherwise Enter ‘1’: "))
        if condition==0:
            import cv2
            import numpy as np

            image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit_fill.png")
            original_image = image.copy()  # Keep a copy of the original image for reset

            # Zoom parameters
            zoom_factor = 1.0
            zoom_step = 0.1
            min_zoom = 0.5
            max_zoom = 3.0

            # Flags and settings
            drawing = False  # True if the mouse is pressed
            erasing = False  # Toggle between drawing and erasing
            brush_size = 2  # Brush size for drawing or erasing
            color = (255, 255, 255)  # Default color (white)
            undo_stack = [image.copy()]  # Stack to store undo history

            # Mouse callback function
            def draw(event, x, y, flags, param):
                global drawing, erasing, image, zoom_factor
                
                # Convert zoomed coordinates back to the original coordinates
                zoom_x = int(x / zoom_factor)
                zoom_y = int(y / zoom_factor)
                
                # Start drawing/erasing when left mouse button is pressed
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    undo_stack.append(image.copy())  # Save the current image state for undo
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                
                # Draw or erase as the mouse moves
                if event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        if not erasing:  # Drawing mode
                            cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                        else:  # Erasing mode
                            cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

            # Function to apply zoom
            def apply_zoom(image, zoom_factor):
                h, w = image.shape[:2]
                zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
                return zoomed_image

            # Trackbar callback function for brush size
            def set_brush_size(val):
                global brush_size
                brush_size = val

            # Trackbar callback function for color selection
            def set_color(val):
                global color
                if val == 0:
                    color = (0, 0, 0)  # Black
                elif val == 1:
                    color = (255, 255, 255)  # White

            # Create a window and set mouse callback
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', draw)

            # Create trackbars for brush size and color selection
            cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
            cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

            while True:
                # Apply zoom to the image before displaying
                zoomed_image = apply_zoom(image, zoom_factor)
                cv2.imshow('Image', zoomed_image)
                
                # Wait for keypress
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('e'):  # Press 'e' to toggle erasing mode
                    erasing = not erasing
                    print('Erasing mode:', erasing)
                
                elif key == ord('r'):  # Press 'r' to reset the image
                    image = original_image.copy()
                    undo_stack.clear()
                    undo_stack.append(image.copy())
                    print('Image reset')
                
                elif key == ord('q'):  # Press 'q' to quit and save the image
                    # Save the image before quitting
                    cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill.png", image)
                    break
                
                elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
                    zoom_factor = min(zoom_factor + zoom_step, max_zoom)
                    print('Zooming in:', zoom_factor)
                
                elif key == ord('-'):  # Press '-' to zoom out
                    zoom_factor = max(zoom_factor - zoom_step, min_zoom)
                    print('Zooming out:', zoom_factor)
                
                elif key == ord('u'):  # Press 'u' for undo
                    if len(undo_stack) > 1:
                        undo_stack.pop()  # Remove the current state
                        image = undo_stack[-1].copy()  # Restore the previous state
                        print('Undo last action')

            # Close the window
            cv2.destroyAllWindows()
            
        else:
            print("vertex neighbours filled with red color")
            
        
        
        
        
        
        
        
        #######################imp extract blue and red (vertex blue and line red which form v shape)
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit_fill.png")
        
        # Convert the image to HSV (Hue, Saturation, Value) for better color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for the red color
        # Red can be tricky as it wraps around in the HSV color space, so we need two ranges
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        
        # Combine both red masks
        red_mask = mask1 + mask2
        
        # Define the lower and upper bounds for the blue color in HSV
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create a mask for the blue range
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        # Combine the red and blue masks
        combined_mask = red_mask + blue_mask
        
        # Create the output image, initializing with black background
        output_image = np.zeros_like(image)
        
        # Apply the combined mask to the original image, keeping only the red and blue pixels
        output_image[combined_mask > 0] = image[combined_mask > 0]
        
        # Save the result
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex.png", output_image)
        
        
        
        
        
        
        
        #############################DETECTING TWO SIDES OF V SHAPE
        
        
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex.png")
        
        # Define color ranges for red, blue, and black in BGR format
        blue_color = np.array([255, 0, 0])  # Blue color in BGR
        red_color = np.array([0, 0, 255])   # Red color in BGR
        
        # Create masks for blue and red
        blue_mask = np.all(image == blue_color, axis=-1)
        red_mask = np.all(image == red_color, axis=-1)
        
        # Directions for 8-connectivity neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        def trace_red_pixels(start_y, start_x, visited):
            queue = [(start_y, start_x)]
            end_pixels = set()
            while queue:
                current_y, current_x = queue.pop(0)
                if (current_y, current_x) not in visited:
                    visited.add((current_y, current_x))
                    is_end_pixel = True
                    # Explore neighbors to find red pixels
                    for dy, dx in directions:
                        ny, nx = current_y + dy, current_x + dx
                        if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:  # Check boundaries
                            if red_mask[ny, nx] and (ny, nx) not in visited:
                                queue.append((ny, nx))
                                is_end_pixel = False
                    if is_end_pixel:
                        end_pixels.add((current_y, current_x))
            return end_pixels
        
        # Iterate through blue pixels to find and trace connected red pixels
        visited = set()
        for blue_y, blue_x in zip(*np.where(blue_mask)):
            # Find all red pixels connected to this blue pixel
            for dy, dx in directions:
                red_y, red_x = blue_y + dy, blue_x + dx
                if 0 <= red_y < image.shape[0] and 0 <= red_x < image.shape[1]:
                    if red_mask[red_y, red_x] and (red_y, red_x) not in visited:
                        # Trace this red line and find all end pixels
                        end_pixels = trace_red_pixels(red_y, red_x, visited)
                        # Mark all end pixels as yellow
                        for end_y, end_x in end_pixels:
                            image[end_y, end_x] = [0, 255, 255]  # Yellow color in BGR
        
        # Save the modified image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex_all.png", image)
        
        
        
        
        
        
        
        
        #################Finding the angel by three points
        ######Angle restriction and average angles
        import cv2
        import numpy as np
        import math
        import pandas as pd

        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex_all.png")

        # Convert image to HSV color space
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for blue and yellow in HSV
        blue_lower = np.array([100, 100, 100])  # Adjust values as per the image
        blue_upper = np.array([140, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        # Create masks for blue and yellow pixels
        blue_mask = cv2.inRange(hsv_img, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

        # Find coordinates for blue and yellow pixels
        blue_coords = np.argwhere(blue_mask > 0)
        yellow_coords = np.argwhere(yellow_mask > 0)

        # Function to calculate the angle between two vectors
        def calculate_angle(p1, p2, vertex):
            v1 = p1 - vertex  # Vector from blue to first yellow
            v2 = p2 - vertex  # Vector from blue to second yellow
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi  # Convert to degrees
            return angle

        # Initialize vertex counter
        vertex_counter = 1

        # Initialize list for angles
        angles = []

        # Define angle restriction range (e.g., between 30 and 150 degrees)
        min_angle = int(input("min angle: "))
        max_angle = int(input("max angle: "))

        # Data storage for Excel file
        angle_details = []

        # Iterate over all blue pixels (vertices)
        for vertex in blue_coords:
            vertex = np.array(vertex[::-1])  # Convert to (x, y) format
            
            # Find the two closest yellow pixels to the blue vertex
            distances_to_yellow = [np.linalg.norm(np.array(y[::-1]) - vertex) for y in yellow_coords]
            if len(distances_to_yellow) < 2:
                continue  # Skip if less than two yellow pixels are found
            
            # Get the indices of the two closest yellow pixels
            closest_yellow_indices = np.argsort(distances_to_yellow)[:2]
            p1 = np.array(yellow_coords[closest_yellow_indices[0]][::-1])
            p2 = np.array(yellow_coords[closest_yellow_indices[1]][::-1])
            
            # Calculate the angle at the blue vertex
            angle = calculate_angle(p1, p2, vertex)
            
            # Check if the angle is within the defined range
            if min_angle <= angle <= max_angle:
                # Draw numbering on the image at the vertex
                cv2.putText(image, str(vertex_counter), tuple(vertex), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Print the angle
                print(f"Vertex {vertex_counter}: Angle = {angle:.2f} degrees")
                
                # Add the angle to the list
                angles.append(angle)
                
                # Store details in the list for Excel
                angle_details.append({"Vertex": vertex_counter, "Coordinates": tuple(vertex), "Angle (degrees)": angle})
                
                vertex_counter += 1

        # Calculate the average angle of all valid angles
        if angles:
            average_angle = np.mean(angles)
            std_dev33=np.std(angles)
        else:
            print("No angles found within the specified range.")

        # Save the results to an Excel file
        angle_df = pd.DataFrame(angle_details)
        angle_df.to_excel(f"{z}_angle_details.xlsx", index=False)

        # Save and display the image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex_all_angle.png", image)

        




        import pandas as pd
        import numpy as np
        from openpyxl import load_workbook

        # Load the Excel file
        file_name = f"{z}_angle_details.xlsx"
        sheet_name = "Sheet1"  # Adjust if the sheet has a different name
        data = pd.read_excel(file_name, sheet_name=sheet_name)

        # Assuming the angles are in a column named 'Angles'
        angle_data = data['Angle (degrees)']

        # Calculate the average and standard deviation
        average_angle33 = np.mean(angle_data)
        std_dev44= np.std(angle_data)

        # Load the workbook and select the sheet
        wb = load_workbook(file_name)
        sheet = wb[sheet_name]

        # Write the results to the Excel file (e.g., in cells D1 and D2)
        sheet["F1"] = "Average Angle "
        sheet["G1"] = f"{average_angle33:.2f} degrees "
        sheet["F2"] = "Standard Deviation"
        sheet["G2"] = f"{std_dev44:.2f} degrees"

        # Save the updated workbook
        wb.save(file_name)

        
        
        print("---------------------------------------------------------------------------------")
        
        
        print("The percentage of tungsten is :",(np.sum(seg2==1)*100)/(np.sum(seg2==1)+np.sum(seg2==0)))
        print("The percentage of Tungsten precipitates is :",(np.sum(seg222==1)*100)/(np.sum(seg222==1)+np.sum(seg222==0)))#true and false 0 ,1
        
        
        print(f"\nAverage grain size: {average_major_diameter:.2f} µm")
        print(f"Minimum Major Grain Diameter: {min_diameter:.2f} µm")
        print(f"Maximum Major Grain Diameter: {max_diameter:.2f} µm")
        print(f"Standard Deviation Grain Diameter: {std_dev:.2f} µm")


        print(f"Average Contiguity : {average_contiguity}")
        print(f"Standard Deviation of Contiguity: {std_dev1:.2f}")


        print(f"Average Angle: {average_angle:.2f} degrees")
        print(f"Standard Deviation of Angles: {std_dev33:.2f} degrees")
    
    
    
    else:
        #######important RED FILLING
        import cv2
        import numpy as np

        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad.png") 

        # Define color ranges (BGR format for OpenCV)
        blue_lower = np.array([100, 0, 0])    # Lower bound for blue
        blue_upper = np.array([255, 50, 50])  # Upper bound for blue
        white_lower = np.array([200, 200, 200])  # Lower bound for white
        white_upper = np.array([255, 255, 255])  # Upper bound for white

        # Find blue pixels
        blue_mask = cv2.inRange(image, blue_lower, blue_upper)

        # Find white pixels
        white_mask = cv2.inRange(image, white_lower, white_upper)

        # Temporary mask to mark red areas
        red_mask = np.zeros_like(blue_mask)

        # Get coordinates of blue pixels
        blue_pixels = np.argwhere(blue_mask > 0)

        # Set a radius of 10 pixels around each blue pixel and find white pixels nearby
        for blue_pixel in blue_pixels:
            y, x = blue_pixel
            # Define the region of interest (ROI) around each blue pixel
            y_min = max(0, y - 6) 
            y_max = min(image.shape[0], y + 6) 
            x_min = max(0, x - 6) 
            x_max = min(image.shape[1], x + 6) 
            
            # Extract the ROI from the white mask
            roi = white_mask[y_min:y_max, x_min:x_max]
            
            # Find coordinates of white pixels in the ROI
            white_pixels_in_roi = np.argwhere(roi > 0)
            
            # Mark white pixels in red_mask with a value of 255 (to signify red)
            for wp in white_pixels_in_roi:
                roi_y, roi_x = wp
                red_mask[y_min + roi_y, x_min + roi_x] += 1

        # Now, fill red in the original image only in locations where red_mask == 1
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if red_mask[y, x] == 1:  # If no overlap, mark as red
                    image[y, x] = [0, 0, 255]  # Red in BGR

        # Save or display the modified image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Red in BGR
        
        # Save the modified image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill.png", image)
        
        condition=int(input("If you wish to make changes Enter ‘0’ otherwise Enter ‘1’: "))
        if condition==0:
            import cv2
            import numpy as np

            image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_fill.png")
            original_image = image.copy()  # Keep a copy of the original image for reset

            # Zoom parameters
            zoom_factor = 1.0
            zoom_step = 0.1
            min_zoom = 0.5
            max_zoom = 3.0

            # Flags and settings
            drawing = False  # True if the mouse is pressed
            erasing = False  # Toggle between drawing and erasing
            brush_size = 2  # Brush size for drawing or erasing
            color = (255, 255, 255)  # Default color (white)
            undo_stack = [image.copy()]  # Stack to store undo history

            # Mouse callback function
            def draw(event, x, y, flags, param):
                global drawing, erasing, image, zoom_factor
                
                # Convert zoomed coordinates back to the original coordinates
                zoom_x = int(x / zoom_factor)
                zoom_y = int(y / zoom_factor)
                
                # Start drawing/erasing when left mouse button is pressed
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    undo_stack.append(image.copy())  # Save the current image state for undo
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                
                # Draw or erase as the mouse moves
                if event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        if not erasing:  # Drawing mode
                            cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                        else:  # Erasing mode
                            cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

            # Function to apply zoom
            def apply_zoom(image, zoom_factor):
                h, w = image.shape[:2]
                zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
                return zoomed_image

            # Trackbar callback function for brush size
            def set_brush_size(val):
                global brush_size
                brush_size = val

            # Trackbar callback function for color selection
            def set_color(val):
                global color
                if val == 0:
                    color = (0, 0, 0)  # Black
                elif val == 1:
                    color = (255, 255, 255)  # White

            # Create a window and set mouse callback
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', draw)

            # Create trackbars for brush size and color selection
            cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
            cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

            while True:
                # Apply zoom to the image before displaying
                zoomed_image = apply_zoom(image, zoom_factor)
                cv2.imshow('Image', zoomed_image)
                
                # Wait for keypress
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('e'):  # Press 'e' to toggle erasing mode
                    erasing = not erasing
                    print('Erasing mode:', erasing)
                
                elif key == ord('r'):  # Press 'r' to reset the image
                    image = original_image.copy()
                    undo_stack.clear()
                    undo_stack.append(image.copy())
                    print('Image reset')
                
                elif key == ord('q'):  # Press 'q' to quit and save the image
                    # Save the image before quitting
                    cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill.png", image)
                    break
                
                elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
                    zoom_factor = min(zoom_factor + zoom_step, max_zoom)
                    print('Zooming in:', zoom_factor)
                
                elif key == ord('-'):  # Press '-' to zoom out
                    zoom_factor = max(zoom_factor - zoom_step, min_zoom)
                    print('Zooming out:', zoom_factor)
                
                elif key == ord('u'):  # Press 'u' for undo
                    if len(undo_stack) > 1:
                        undo_stack.pop()  # Remove the current state
                        image = undo_stack[-1].copy()  # Restore the previous state
                        print('Undo last action')

            # Close the window
            cv2.destroyAllWindows()
        
        else:
            print("vertex neighbours filled with red color")
        
        
        
        
        
        
        
        #######################imp extract blue and red (vertex blue and line red which form v shape)
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_fill.png")
        
        # Convert the image to HSV (Hue, Saturation, Value) for better color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for the red color
        # Red can be tricky as it wraps around in the HSV color space, so we need two ranges
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        
        # Combine both red masks
        red_mask = mask1 + mask2
        
        # Define the lower and upper bounds for the blue color in HSV
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create a mask for the blue range
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        # Combine the red and blue masks
        combined_mask = red_mask + blue_mask
        
        # Create the output image, initializing with black background
        output_image = np.zeros_like(image)
        
        # Apply the combined mask to the original image, keeping only the red and blue pixels
        output_image[combined_mask > 0] = image[combined_mask > 0]
        
        # Save the result
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill_ex.png", output_image)
        
        
        
        
        
        
        
        #############################DETECTING TWO SIDES OF V SHAPE
        
        
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_fill_ex.png")
        
        # Define color ranges for red, blue, and black in BGR format
        blue_color = np.array([255, 0, 0])  # Blue color in BGR
        red_color = np.array([0, 0, 255])   # Red color in BGR
        
        # Create masks for blue and red
        blue_mask = np.all(image == blue_color, axis=-1)
        red_mask = np.all(image == red_color, axis=-1)
        
        # Directions for 8-connectivity neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        def trace_red_pixels(start_y, start_x, visited):
            queue = [(start_y, start_x)]
            end_pixels = set()
            while queue:
                current_y, current_x = queue.pop(0)
                if (current_y, current_x) not in visited:
                    visited.add((current_y, current_x))
                    is_end_pixel = True
                    # Explore neighbors to find red pixels
                    for dy, dx in directions:
                        ny, nx = current_y + dy, current_x + dx
                        if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:  # Check boundaries
                            if red_mask[ny, nx] and (ny, nx) not in visited:
                                queue.append((ny, nx))
                                is_end_pixel = False
                    if is_end_pixel:
                        end_pixels.add((current_y, current_x))
            return end_pixels
        
        # Iterate through blue pixels to find and trace connected red pixels
        visited = set()
        for blue_y, blue_x in zip(*np.where(blue_mask)):
            # Find all red pixels connected to this blue pixel
            for dy, dx in directions:
                red_y, red_x = blue_y + dy, blue_x + dx
                if 0 <= red_y < image.shape[0] and 0 <= red_x < image.shape[1]:
                    if red_mask[red_y, red_x] and (red_y, red_x) not in visited:
                        # Trace this red line and find all end pixels
                        end_pixels = trace_red_pixels(red_y, red_x, visited)
                        # Mark all end pixels as yellow
                        for end_y, end_x in end_pixels:
                            image[end_y, end_x] = [0, 255, 255]  # Yellow color in BGR
        
        # Save the modified image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill_ex_all.png", image)
        
        
        
        
        
        
        
        
        #################Finding the angel by three points
        ######Angle restriction and average angles
        import cv2
        import numpy as np
        import math
        import pandas as pd

        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_fill_ex_all.png")

        # Convert image to HSV color space
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for blue and yellow in HSV
        blue_lower = np.array([100, 100, 100])  # Adjust values as per the image
        blue_upper = np.array([140, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        # Create masks for blue and yellow pixels
        blue_mask = cv2.inRange(hsv_img, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

        # Find coordinates for blue and yellow pixels
        blue_coords = np.argwhere(blue_mask > 0)
        yellow_coords = np.argwhere(yellow_mask > 0)

        # Function to calculate the angle between two vectors
        def calculate_angle(p1, p2, vertex):
            v1 = p1 - vertex  # Vector from blue to first yellow
            v2 = p2 - vertex  # Vector from blue to second yellow
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi  # Convert to degrees
            return angle

        # Initialize vertex counter
        vertex_counter = 1

        # Initialize list for angles
        angles = []

        # Define angle restriction range (e.g., between 30 and 150 degrees)
        min_angle = int(input("min angle: "))
        max_angle = int(input("max angle: "))

        # Data storage for Excel file
        angle_details = []

        # Iterate over all blue pixels (vertices)
        for vertex in blue_coords:
            vertex = np.array(vertex[::-1])  # Convert to (x, y) format
            
            # Find the two closest yellow pixels to the blue vertex
            distances_to_yellow = [np.linalg.norm(np.array(y[::-1]) - vertex) for y in yellow_coords]
            if len(distances_to_yellow) < 2:
                continue  # Skip if less than two yellow pixels are found
            
            # Get the indices of the two closest yellow pixels
            closest_yellow_indices = np.argsort(distances_to_yellow)[:2]
            p1 = np.array(yellow_coords[closest_yellow_indices[0]][::-1])
            p2 = np.array(yellow_coords[closest_yellow_indices[1]][::-1])
            
            # Calculate the angle at the blue vertex
            angle = calculate_angle(p1, p2, vertex)
            
            # Check if the angle is within the defined range
            if min_angle <= angle <= max_angle:
                # Draw numbering on the image at the vertex
                cv2.putText(image, str(vertex_counter), tuple(vertex), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Print the angle
                print(f"Vertex {vertex_counter}: Angle = {angle:.2f} degrees")
                
                # Add the angle to the list
                angles.append(angle)
                
                # Store details in the list for Excel
                angle_details.append({"Vertex": vertex_counter, "Coordinates": tuple(vertex), "Angle (degrees)": angle})
                
                vertex_counter += 1

        # Calculate the average angle of all valid angles
        if angles:
            average_angle = np.mean(angles)
            std_dev33=np.std(angles)
        else:
            print("No angles found within the specified range.")

        # Save the results to an Excel file
        angle_df = pd.DataFrame(angle_details)
        angle_df.to_excel(f"{z}_angle_details.xlsx", index=False)

        # Save and display the image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill_ex_all_angle.png", image)

       




        import pandas as pd
        import numpy as np
        from openpyxl import load_workbook

        # Load the Excel file
        file_name = f"{z}_angle_details.xlsx"
        sheet_name = "Sheet1"  # Adjust if the sheet has a different name
        data = pd.read_excel(file_name, sheet_name=sheet_name)

        # Assuming the angles are in a column named 'Angles'
        angle_data = data['Angle (degrees)']

        # Calculate the average and standard deviation
        average_angle33 = np.mean(angle_data)
        std_dev44= np.std(angle_data)

        # Load the workbook and select the sheet
        wb = load_workbook(file_name)
        sheet = wb[sheet_name]

        # Write the results to the Excel file (e.g., in cells D1 and D2)
        sheet["F1"] = "Average Angle "
        sheet["G1"] = f"{average_angle33:.2f} degrees "
        sheet["F2"] = "Standard Deviation"
        sheet["G2"] = f"{std_dev44:.2f} degrees"

        # Save the updated workbook
        wb.save(file_name)

        
        
        
        
        
        print("---------------------------------------------------------------------------------")
        
        
        print("The percentage of tungsten is :",(np.sum(seg2==1)*100)/(np.sum(seg2==1)+np.sum(seg2==0)))
        print("The percentage of Tungsten precipitates is :",(np.sum(seg222==1)*100)/(np.sum(seg222==1)+np.sum(seg222==0)))#true and false 0 ,1
        
        print(f"\nAverage grain size : {average_major_diameter:.2f} µm")
        print(f"Minimum Major Grain Diameter: {min_diameter:.2f} µm")
        print(f"Maximum Major Grain Diameter: {max_diameter:.2f} µm")
        print(f"Standard Deviation Grain Diameter: {std_dev:.2f} µm")


        print(f"Average Contiguity : {average_contiguity}")
        print(f"Standard Deviation of Contiguity: {std_dev1:.2f}")


        print(f"Average Angle: {average_angle:.2f} degrees")
        print(f"Standard Deviation of Angles: {std_dev33:.2f} degrees")






else:
    print("Please clean the threshold image")
    
    
    
    import cv2
    import numpy as np

    def filter_black_clusters(image_path, min_pixels, max_pixels):
        # Step 1: Load the image (grayscale)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Step 2: Threshold the image to ensure it's binary (0 = black, 255 = white)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Step 3: Find connected components (black pixels are connected)
        num_labels, labels_im = cv2.connectedComponents(binary_image)

        # Step 4: Filter the connected components based on the pixel count range
        for label in range(1, num_labels):  # Start from 1 because 0 is the background
            # Get all the pixels that belong to this component
            component = (labels_im == label).astype(np.uint8)

            # Count the number of black pixels in this component
            pixel_size = np.sum(component)

            # Step 5: If the component size is within the specified range, color it white on the original image
            if min_pixels <= pixel_size <= max_pixels:
                image[component == 1] = 255  # Set those pixels to white in the original image

        # Step 6: Save and display the modified image
        cv2.imwrite(output_path, image)

    # Usage example
    image_path = f"{z}_thresh.png"  # Path to the input image
    output_path= f"{z}_thresh.png"
    filter_black_clusters(image_path, 1, 10)
    
    
    
    
    import cv2
    import numpy as np

    image = cv2.imread(f"{z}_thresh.png")
    original_image = image.copy()  # Keep a copy of the original image for reset

    # Zoom parameters
    zoom_factor = 1.0
    zoom_step = 0.1
    min_zoom = 0.5
    max_zoom = 3.0

    # Flags and settings
    drawing = False  # True if the mouse is pressed
    erasing = False  # Toggle between drawing and erasing
    brush_size = 2  # Brush size for drawing or erasing
    color = (255, 255, 255)  # Default color (white)
    undo_stack = [image.copy()]  # Stack to store undo history

    # Mouse callback function
    def draw(event, x, y, flags, param):
        global drawing, erasing, image, zoom_factor
        
        # Convert zoomed coordinates back to the original coordinates
        zoom_x = int(x / zoom_factor)
        zoom_y = int(y / zoom_factor)
        
        # Start drawing/erasing when left mouse button is pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            undo_stack.append(image.copy())  # Save the current image state for undo
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        
        # Draw or erase as the mouse moves
        if event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if not erasing:  # Drawing mode
                    cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                else:  # Erasing mode
                    cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

    # Function to apply zoom
    def apply_zoom(image, zoom_factor):
        h, w = image.shape[:2]
        zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
        return zoomed_image

    # Trackbar callback function for brush size
    def set_brush_size(val):
        global brush_size
        brush_size = val

    # Trackbar callback function for color selection
    def set_color(val):
        global color
        if val == 0:
            color = (0, 0, 0)  # Black
        elif val == 1:
            color = (255, 255, 255)  # White

    # Create a window and set mouse callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw)

    # Create trackbars for brush size and color selection
    cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
    cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

    while True:
        # Apply zoom to the image before displaying
        zoomed_image = apply_zoom(image, zoom_factor)
        cv2.imshow('Image', zoomed_image)
        
        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('e'):  # Press 'e' to toggle erasing mode
            erasing = not erasing
            print('Erasing mode:', erasing)
        
        elif key == ord('r'):  # Press 'r' to reset the image
            image = original_image.copy()
            undo_stack.clear()
            undo_stack.append(image.copy())
            print('Image reset')
        
        elif key == ord('q'):  # Press 'q' to quit and save the image
            # Save the image before quitting
            cv2.imwrite(f"{z}_thresh_cleaned.png", image)
            break
        
        elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
            zoom_factor = min(zoom_factor + zoom_step, max_zoom)
            print('Zooming in:', zoom_factor)
        
        elif key == ord('-'):  # Press '-' to zoom out
            zoom_factor = max(zoom_factor - zoom_step, min_zoom)
            print('Zooming out:', zoom_factor)
        
        elif key == ord('u'):  # Press 'u' for undo
            if len(undo_stack) > 1:
                undo_stack.pop()  # Remove the current state
                image = undo_stack[-1].copy()  # Restore the previous state
                print('Undo last action')

    # Close the window
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    #resize
    img=cv2.imread(f"{z}_thresh_cleaned.png")
    resize=cv2.resize(img, (512,512))
    cv2.imwrite(f"{z}_thresh_cleaned_512.png", resize)
    
    #resize
    img=cv2.imread(f"{z}_thresh_cleaned.png")
    resize=cv2.resize(img, (256,256))
    cv2.imwrite(f"{z}_thresh_cleaned_256.png", resize)
    
    #prediction    
    import cv2
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
    from keras.utils import normalize
    from matplotlib import pyplot as plt

    # Function to build the U-Net model
    def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = inputs
        
        # Contracting path
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def process_image(model, input_image_path, img_size):
        input_image = cv2.imread(input_image_path, 0)
        input_image_norm = normalize(np.array(input_image), axis=1)
        input_image_resized = cv2.resize(input_image_norm, (img_size, img_size))  # Resize to fit the model input
        input_image_resized = np.expand_dims(np.expand_dims(input_image_resized, axis=0), axis=3)

        # Predict
        prediction = (model.predict(input_image_resized)[0, :, :, 0] > 0.2).astype(np.uint8)
        return prediction

    # Define model and image paths
    model_paths = ['high_mag_e180_256vg.hdf5', 'low_mag_e300_256vg.hdf5']
    input_image_paths = [f"{z}_thresh_cleaned_256.png", f"{z}_thresh_cleaned_512.png"]
    img_sizes = [256, 512]

    # Load and predict for all images
    predictions = []
    for model_path, img_size, input_image_path in zip(model_paths * 2, img_sizes * 2, input_image_paths * 2):
        model = simple_unet_model(img_size, img_size, 1)
        model.load_weights(model_path)
        prediction = process_image(model, input_image_path, img_size)
        predictions.append(prediction)

    # Display all predictions with matplotlib
    plt.figure(figsize=(12, 12))
    for i, pred in enumerate(predictions):
        plt.subplot(2, 2, i + 1)
        plt.title(f'Prediction {i+1}')
        plt.imshow(pred, cmap='gray')
    plt.show()

    # Prompt for saving the selected prediction
    save_choice = int(input("Enter the prediction number (1-4) you want to save: ")) - 1
    if 0 <= save_choice < 4:
        cv2.imwrite(f"{z}_thresh_cleaned_resize_b.png", predictions[save_choice] * 255)
        print(f"Prediction {save_choice + 1} saved successfully.")
    else:
        sys.exit()
    
    
    
    
    #resize
    img=cv2.imread(f"{z}_thresh_cleaned_resize_b.png")
    resize=cv2.resize(img, (thresh.shape[1],thresh.shape[0]))
    cv2.imwrite(f"{z}_thresh_cleaned_resize_b_resize.png", resize)
    
    
    import cv2
    import numpy as np

    image = cv2.imread(f"{z}_thresh_cleaned_resize_b_resize.png")
    original_image = image.copy()  # Keep a copy of the original image for reset

    # Zoom parameters
    zoom_factor = 1.0
    zoom_step = 0.1
    min_zoom = 0.5
    max_zoom = 3.0

    # Flags and settings
    drawing = False  # True if the mouse is pressed
    erasing = False  # Toggle between drawing and erasing
    brush_size = 2  # Brush size for drawing or erasing
    color = (255, 255, 255)  # Default color (white)
    undo_stack = [image.copy()]  # Stack to store undo history

    # Mouse callback function
    def draw(event, x, y, flags, param):
        global drawing, erasing, image, zoom_factor
        
        # Convert zoomed coordinates back to the original coordinates
        zoom_x = int(x / zoom_factor)
        zoom_y = int(y / zoom_factor)
        
        # Start drawing/erasing when left mouse button is pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            undo_stack.append(image.copy())  # Save the current image state for undo
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        
        # Draw or erase as the mouse moves
        if event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if not erasing:  # Drawing mode
                    cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                else:  # Erasing mode
                    cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

    # Function to apply zoom
    def apply_zoom(image, zoom_factor):
        h, w = image.shape[:2]
        zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
        return zoomed_image

    # Trackbar callback function for brush size
    def set_brush_size(val):
        global brush_size
        brush_size = val

    # Trackbar callback function for color selection
    def set_color(val):
        global color
        if val == 0:
            color = (0, 0, 0)  # Black
        elif val == 1:
            color = (255, 255, 255)  # White

    # Create a window and set mouse callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw)

    # Create trackbars for brush size and color selection
    cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
    cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

    while True:
        # Apply zoom to the image before displaying
        zoomed_image = apply_zoom(image, zoom_factor)
        cv2.imshow('Image', zoomed_image)
        
        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('e'):  # Press 'e' to toggle erasing mode
            erasing = not erasing
            print('Erasing mode:', erasing)
        
        elif key == ord('r'):  # Press 'r' to reset the image
            image = original_image.copy()
            undo_stack.clear()
            undo_stack.append(image.copy())
            print('Image reset')
        
        elif key == ord('q'):  # Press 'q' to quit and save the image
            # Save the image before quitting
            cv2.imwrite(f"{z}_thresh_cleaned_resize_b_resize.png", image)
            break
        
        elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
            zoom_factor = min(zoom_factor + zoom_step, max_zoom)
            print('Zooming in:', zoom_factor)
        
        elif key == ord('-'):  # Press '-' to zoom out
            zoom_factor = max(zoom_factor - zoom_step, min_zoom)
            print('Zooming out:', zoom_factor)
        
        elif key == ord('u'):  # Press 'u' for undo
            if len(undo_stack) > 1:
                undo_stack.pop()  # Remove the current state
                image = undo_stack[-1].copy()  # Restore the previous state
                print('Undo last action')

    # Close the window
    cv2.destroyAllWindows()
    
    
    #segmentation
    img=cv2.imread(f"{z}_thresh_cleaned.png",0)
    seg1=img==0
    seg2=img==255
    allsegment=np.zeros((img.shape[0],img.shape[1],3))
    allsegment[seg1]=(1,0,0)
    allsegment[seg2]=(0,1,0)
    plt.imsave(f"{z}_thresh_cleaned_seg.png",allsegment)
    
    print("The percentage of tungsten is :",(np.sum(seg2==1)*100)/(np.sum(seg2==1)+np.sum(seg2==0)))#true and false 0 ,1
    
    
    
    
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage import img_as_ubyte

    # Load the image in grayscale mode
    image = cv2.imread(f"{z}_thresh_cleaned_resize_b_resize.png", cv2.IMREAD_GRAYSCALE)

    # Ensure the white boundary is represented as 255 and the background as 0
    # If your white boundary is not pure white (255), you may need to threshold it
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological erosion to thin the boundary
    # The kernel defines the neighborhood size for erosion, adjust if necessary
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=0)

    # Skeletonize the image to ensure the boundary is thinned
    skeleton = skeletonize(eroded_image // 255)  # Skeletonize expects a binary image (0, 1)
    skeleton_image = img_as_ubyte(skeleton)  # Convert back to 0-255 range

    # Dilate the skeleton to ensure uniform 2-pixel thickness boundary
    thinned_image = cv2.dilate(skeleton_image, kernel, iterations=0)

    # Save the output image

    # Display the result (optional)

    cv2.imwrite(f"{z}_thresh_cleaned_resize_b_resize_thin.png", thinned_image)



    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_thresh_cleaned_seg.png")
    boundaryImg = cv2.imread(f"{z}_thresh_cleaned_resize_b_resize_thin.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [255, 255, 255]
    # Display the merged image




    cv2.imwrite(f"{z}_merged_thin.png", outputImg)





    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_merged_thin.png")

    # Define pixel values for green, white, and black (background)
    green_value = [0, 255, 0]  # Adjust if needed (BGR format)
    white_value = [255, 255, 255]  # White for boundary
    black_value = [0, 0, 0]  # Black for background

    # Convert the image to grayscale for easier processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask for white pixels
    white_mask = cv2.inRange(image, np.array(white_value), np.array(white_value))

    # Create a mask for green pixels
    green_mask = cv2.inRange(image, np.array(green_value), np.array(green_value))

    # Create an output image initialized as black
    output_image = np.zeros_like(gray_image)

    # Get the height and width of the image
    height, width = gray_image.shape

    # Define a function to check if all 8 pixels are green in a specific direction
    def check_green_complete(image, x, y, direction):
        if direction == 'vertical':
            # Check 8 pixels above and 8 pixels below
            if y - 7 >= 0 and y + 7 < height:
                return np.all(image[y-7:y, x] == 255) and np.all(image[y+1:y+8, x] == 255)
        elif direction == 'horizontal':
            # Check 8 pixels left and 8 pixels right
            if x - 7 >= 0 and x + 7 < width:
                return np.all(image[y, x-7:x] == 255) and np.all(image[y, x+1:x+7] == 255)
        return False

    # Iterate over each white pixel
    for y in range(height):
        for x in range(width):
            if white_mask[y, x] == 255:  # If the pixel is white
                # Check vertical and horizontal conditions
                if check_green_complete(green_mask, x, y, 'vertical') or check_green_complete(green_mask, x, y, 'horizontal'):
                    # Set the output pixel to white if conditions are satisfied
                    output_image[y, x] = 255

    # Save the output image
    cv2.imwrite(f"{z}_merged_thin_ex.png", output_image)







    import cv2
    import numpy as np
    from skimage.morphology import disk, dilation
    from skimage import io, img_as_ubyte
    from skimage.color import rgb2gray
    from skimage.feature import canny
    
    # Read the original image
    original_image = cv2.imread(f"{z}_thresh_cleaned_seg.png")
    
    # Convert to grayscale
    gray_image = rgb2gray(original_image)
    
    # Detect edges using the Canny method
    edge_image = canny(gray_image)
    
    # Define a structuring element for dilation (disk of radius 0)
    selem = disk(0)
    
    # Dilate the edges
    dilated_edges = dilation(edge_image, selem)
    
    # Convert the dilated edge image to uint8 type for proper overlay
    dilated_edges_uint8 = img_as_ubyte(dilated_edges)
    
    # Overlay the dilated edges on the original image (in white)
    overlayed_image = original_image.copy()
    overlayed_image[dilated_edges_uint8 > 0] = [255, 255, 255]
    
    # Display the dilated edges
    
    
    # Save the result image
    cv2.imwrite(f"{z}_perimeter.png", overlayed_image)






    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_perimeter.png")
    boundaryImg = cv2.imread(f"{z}_merged_thin_ex.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [255, 255, 255]
    # Display the merged image



    cv2.imwrite(f"{z}_merged_thin_img.png", outputImg)
    
    
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage import img_as_ubyte

    # Load the image in grayscale mode
    image = cv2.imread(f"{z}_thresh_cleaned_resize_b_resize.png", cv2.IMREAD_GRAYSCALE)

    # Ensure the white boundary is represented as 255 and the background as 0
    # If your white boundary is not pure white (255), you may need to threshold it
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological erosion to thin the boundary
    # The kernel defines the neighborhood size for erosion, adjust if necessary
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=0)

    # Skeletonize the image to ensure the boundary is thinned
    skeleton = skeletonize(eroded_image // 255)  # Skeletonize expects a binary image (0, 1)
    skeleton_image = img_as_ubyte(skeleton)  # Convert back to 0-255 range

    # Dilate the skeleton to ensure uniform 2-pixel thickness boundary
    thinned_image = cv2.dilate(skeleton_image, kernel, iterations=1)

    # Save the output image

    # Display the result (optional)
    #cv2.imshow('Thinned Boundary Image', thinned_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    cv2.imwrite(f"{z}_thresh_cleaned_resize_b_resize_thick2pix.png", thinned_image)



    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_thresh_cleaned_seg.png")
    boundaryImg = cv2.imread(f"{z}_thresh_cleaned_resize_b_resize_thick2pix.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [255, 255, 255]
    # Display the merged image

    #cv2.imshow('Merged Image', outputImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save the merged image if needed
    cv2.imwrite(f"{z}_merge_thick2pix.png", outputImg)
    
    
    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_thresh_cleaned_seg.png")
    boundaryImg = cv2.imread(f"{z}_thresh_cleaned_resize_b_resize_thick2pix.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [0, 0, 0]
    # Display the merged image

    #cv2.imshow('Merged Image', outputImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save the merged image if needed
    cv2.imwrite(f"{z}_merge_thick2pix_black_b.png", outputImg)
    
    
    
    import cv2
    import numpy as np

    #image = cv2.imread(f"{z}_merge_thick2pix.png")
    #original_image = image.copy()  # Keep a copy of the original image for reset

    # Zoom parameters
    #zoom_factor = 1.0
   # zoom_step = 0.1
    #min_zoom = 0.5
   # max_zoom = 3.0

    # Flags and settings
    #drawing = False  # True if the mouse is pressed
    #erasing = False  # Toggle between drawing and erasing
    #brush_size = 2  # Brush size for drawing or erasing
    #color = (255, 255, 255)  # Default color (white)
    #undo_stack = [image.copy()]  # Stack to store undo history

    # Mouse callback function
    #def draw(event, x, y, flags, param):
       # global drawing, erasing, image, zoom_factor
        
        # Convert zoomed coordinates back to the original coordinates
       # zoom_x = int(x / zoom_factor)
        #zoom_y = int(y / zoom_factor)
        
        # Start drawing/erasing when left mouse button is pressed
        #if event == cv2.EVENT_LBUTTONDOWN:
            #drawing = True
            #undo_stack.append(image.copy())  # Save the current image state for undo
        #elif event == cv2.EVENT_LBUTTONUP:
            #drawing = False
        
        # Draw or erase as the mouse moves
        #if event == cv2.EVENT_MOUSEMOVE:
            #if drawing:
                #if not erasing:  # Drawing mode
                    #cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
               # else:  # Erasing mode
                    #cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

    # Function to apply zoom
    #def apply_zoom(image, zoom_factor):
        #h, w = image.shape[:2]
        #zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
        #return zoomed_image

    # Trackbar callback function for brush size
   # def set_brush_size(val):
       # global brush_size
        #brush_size = val

    # Trackbar callback function for color selection
    #def set_color(val):
       # global color
        #if val == 0:
           # color = (255, 0, 0)  # Blue
       # elif val == 1:
            #color = (255, 255, 255)  # White

    # Create a window and set mouse callback
   # cv2.namedWindow('Image')
    #cv2.setMouseCallback('Image', draw)

    # Create trackbars for brush size and color selection
    #cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
    #cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

    #while True:
        # Apply zoom to the image before displaying
        #zoomed_image = apply_zoom(image, zoom_factor)
        #cv2.imshow('Image', zoomed_image)
        
        # Wait for keypress
        #key = cv2.waitKey(1) & 0xFF
        
        #if key == ord('e'):  # Press 'e' to toggle erasing mode
           # erasing = not erasing
            #print('Erasing mode:', erasing)
        
       # elif key == ord('r'):  # Press 'r' to reset the image
            #image = original_image.copy()
            #undo_stack.clear()
            #undo_stack.append(image.copy())
            #print('Image reset')
        
        #elif key == ord('q'):  # Press 'q' to quit and save the image
            # Save the image before quitting
           # cv2.imwrite(f"{z}_merge_thick2pix.png", image)
            #break
        
       # elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
            #zoom_factor = min(zoom_factor + zoom_step, max_zoom)
            #print('Zooming in:', zoom_factor)
        
       # elif key == ord('-'):  # Press '-' to zoom out
            #zoom_factor = max(zoom_factor - zoom_step, min_zoom)
            #print('Zooming out:', zoom_factor)
        
       # elif key == ord('u'):  # Press 'u' for undo
           # if len(undo_stack) > 1:
             #   undo_stack.pop()  # Remove the current state
               # image = undo_stack[-1].copy()  # Restore the previous state
               # print('Undo last action')

    # Close the window
    #cv2.destroyAllWindows()

    
    
    
    
    
    import cv2
    import numpy as np

    # Step 1: Load the image
    image = cv2.imread(f"{z}_merge_thick2pix.png")

    # Step 2: Separate the channels
    green_channel = image[:, :, 1]  # Extract the green channel
    red_channel = image[:, :, 2]  # Extract the red channel

    # Step 3: Create a mask for the white boundary
    lower_white = np.array([250, 250, 250])  # Lower bound for white
    upper_white = np.array([255, 255, 255])  # Upper bound for white
    white_mask = cv2.inRange(image, lower_white, upper_white)

    # Step 4: Find contours of the white boundary
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Create a mask for enclosed green regions
    green_enclosed_mask = np.zeros_like(green_channel, dtype=np.uint8)

    for contour in contours:
        # Create a blank mask for each contour
        contour_mask = np.zeros_like(green_channel, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)  # Fill the contour
        
        # Check if green pixels are within the contour
        green_in_contour = cv2.bitwise_and(green_channel, contour_mask)
        if np.any(green_in_contour > 0):  # If green is present inside the contour
            green_enclosed_mask = cv2.bitwise_or(green_enclosed_mask, green_in_contour)

    # Step 6: Combine the green enclosed regions with the white boundary
    output_image = np.zeros_like(image)  # Start with a black image
    output_image[:, :, 1] = green_enclosed_mask  # Add the enclosed green regions
    output_image[:, :, :] = cv2.bitwise_or(output_image, cv2.merge([white_mask, white_mask, white_mask]))  # Add the white boundary

    # Step 7: Save or display the result
    cv2.imwrite(f"{z}_enclosed_green_with_boundary.png", output_image)
    #cv2.imshow('Enclosed Green Regions with White Boundary', output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_enclosed_green_with_boundary.png")

    # Convert the image from BGR to HSV color space for easier color thresholding
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the green color in HSV
    lower_green = np.array([35, 100, 100])  # Adjust these values based on your image
    upper_green = np.array([85, 255, 255])

    # Create a mask for green pixels
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Create an output image where everything else is black
    output = np.zeros_like(image)
    output[mask > 0] = image[mask > 0]

    # Convert the non-green pixels to black
    output[mask == 0] = [0, 0, 0]  # Background becomes black

    # Save and display the result
    cv2.imwrite(f"{z}_enclosed_green_with_boundary_ex_green.png", output)
   # cv2.imshow('Green Pixels', output)
    #cv2.waitKey(0)
   # cv2.destroyAllWindows()



    import cv2
    import numpy as np

    def filter_green_clusters(image_path, min_pixels, max_pixels):
        # Step 1: Load the image in color
        image = cv2.imread(image_path)
        
        # Step 2: Create a mask for green pixels (threshold for green)
        # Define the lower and upper bounds for the green color in BGR format
        lower_green = np.array([0, 100, 0])  # Lower bound for green (BGR)
        upper_green = np.array([100, 255, 100])  # Upper bound for green (BGR)

        # Threshold the image to get only green pixels
        green_mask = cv2.inRange(image, lower_green, upper_green)

        # Step 3: Find connected components for green pixels
        num_labels, labels_im = cv2.connectedComponents(green_mask)

        # Create an output image, initially all black
        output_image = np.zeros_like(green_mask)

        # Step 4: Filter the connected components based on the pixel count range
        for label in range(1, num_labels):  # Start from 1 because 0 is the background
            # Get all the pixels that belong to this component
            component = (labels_im == label).astype(np.uint8)

            # Count the number of green pixels in this component
            pixel_size = np.sum(component)

            # Step 5: If the component size is within the specified range, retain it
            if min_pixels <= pixel_size <= max_pixels:
                output_image[component == 1] = 255  # Set pixels of this component to white

        # Step 6: Save and display the output image
        cv2.imwrite(f"{z}_enclosed_binary.png", output_image)
        #cv2.imshow('Filtered Green Clusters', output_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    # Usage example
    image_path = f"{z}_enclosed_green_with_boundary_ex_green.png"  # Path to the input image
    filter_green_clusters(image_path, 7, 1000000000)



    import matplotlib.pyplot as plt

    img=cv2.imread(f"{z}_enclosed_binary.png",0)
    seg_1=img==0
    seg_2=img==255
    allsegment=np.zeros((img.shape[0],img.shape[1],3))
    allsegment[seg_1]=(1,0,0)
    allsegment[seg_2]=(0,1,0)
    plt.imsave(f"{z}_enclosed_binary_seg.png",allsegment)




    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Load the image
    image = cv2.imread(f"{z}_enclosed_binary_seg.png")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Threshold the image to isolate green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours of the green regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image for drawing
    output_image = image.copy()

    # Get scale information
    #croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
    #croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
    #conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel

    # Initialize variables
    valid_contours = 0
    total_major_diameter = 0
    contour_data = []
    major_diameters = []

    # Process contours
    for i, contour in enumerate(contours, 1):
        if cv2.contourArea(contour) < 1:  # Skip small contours
            continue

        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cX = int(moments['m10'] / moments['m00'])
            cY = int(moments['m01'] / moments['m00'])

            # Bounding box and major diameter
            x, y, w, h = cv2.boundingRect(contour)
            major_diameter = max(w, h) * conversion_factor  # µm
            area_um2 = cv2.contourArea(contour) * conversion_factor ** 2
            perimeter_um = cv2.arcLength(contour, True) * conversion_factor
            particle_edge_um = area_um2 ** 0.5
            circularity_um = 4 * 3.14 * area_um2 / (perimeter_um ** 2)

            # Update variables
            total_major_diameter += major_diameter
            major_diameters.append(major_diameter)
            contour_data.append((contour, cX, cY, major_diameter))
            valid_contours += 1

            # Output details
            print(f'Region {valid_contours}:')
            print(f'  Area: {area_um2:.2f} µm²')
            print(f'  Perimeter: {perimeter_um:.2f} µm')
            print(f'  Major Diameter: {major_diameter:.2f} µm')
            print(f'  Particle Edge: {particle_edge_um:.2f} µm')
            print(f'  Circularity: {circularity_um:.2f} µm')
            print()

    # Final calculations
    if valid_contours > 0:
        average_major_diameter = total_major_diameter / valid_contours
        min_diameter = min(major_diameters)
        max_diameter = max(major_diameters)
        # Calculate the standard deviation using numpy
        std_dev = np.std(major_diameters)
        print(f"\nAverage Major Diameter: {average_major_diameter:.2f} µm")
        print(f"Minimum Major Diameter: {min(major_diameters):.2f} µm")
        print(f"Maximum Major Diameter: {max(major_diameters):.2f} µm")
        print(f"Standard Deviation: {np.std(major_diameters):.2f} µm")

    # Draw contours and label numbers
    for i, (contour, cX, cY, major_diameter) in enumerate(contour_data, 1):
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        text = str(i)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        textX = cX - text_size[0] // 2
        textY = cY + text_size[1] // 2
        cv2.putText(output_image, text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save and display results
    cv2.imwrite(f"{z}_grain_regions1.png", output_image)
   # cv2.imshow("Labeled Image", output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Particle size distribution
    if major_diameters:
        major_diameters = np.array(major_diameters)
        hist, bins = np.histogram(major_diameters, bins='auto')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        mu, std = norm.fit(major_diameters)

        plt.bar(bin_centers, hist, width=(bins[1] - bins[0]), alpha=0.6, color='purple', edgecolor='black', linewidth=1.0)
        x = np.linspace(bin_centers[0], bin_centers[-1], 100)
        p = norm.pdf(x, mu, std) * len(major_diameters) * (bins[1] - bins[0])
        plt.plot(x, p, 'b', linewidth=2)
        plt.title('W Particle Size Distribution')
        plt.xlabel('W Particle Size (µm)')
        plt.ylabel('Count')
        plt.xlim([0, max(major_diameters) + 10])
        plt.ylim([0, max(hist) + 5])
        plt.savefig(f"{z}_Particle_Size_distribution1.png", dpi=300, bbox_inches='tight')
        plt.show()




    import pandas as pd

    # Initialize a list to store the details of each valid region
    region_details = []
    valid_contours = 0  # Counter for valid regions

    # Process each contour and store the details
    for contour in contours:
        if cv2.contourArea(contour) < 1:  # Ignore small contours
            continue

        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            x, y, w, h = cv2.boundingRect(contour)
            major_diameter = max(w, h) * conversion_factor  # Major diameter in µm
            area_um2 = cv2.contourArea(contour) * conversion_factor ** 2
            perimeter_um = cv2.arcLength(contour, True) * conversion_factor
            particle_edge_um = area_um2 ** 0.5  # Corrected to compute the square root of area_um2
            circularity_um = 4 * 3.14 * area_um2 / (perimeter_um ** 2)  # Corrected to square the perimeter_um variable

            # Increment valid region counter
            valid_contours += 1

            # Append details to the list
            region_details.append({
                'Region': valid_contours,  # Use valid_contours as region index
                'Area (µm²)': round(area_um2, 2),
                'Perimeter (µm)': round(perimeter_um, 2),
                'Major Diameter (µm)': round(major_diameter, 2),
                'Particle Edge (µm)': round(particle_edge_um, 2),
                'Circularity (µm)': round(circularity_um, 2)
            })

    # Ensure all valid regions match the image labels
    if len(region_details) != valid_contours:
        raise ValueError("Mismatch between the number of labeled regions and saved region details.")

    # Create a pandas DataFrame from the region details
    df = pd.DataFrame(region_details)

    # Save the DataFrame to an Excel file
    output_file = f"{z}_region_details.xlsx"
    df.to_excel(output_file, index=False)

    print(f"Region details successfully saved to {output_file}")




    import pandas as pd
    import numpy as np
    from openpyxl import load_workbook

    # Load the Excel file
    file_name = f"{z}_region_details.xlsx"
    sheet_name = "Sheet1"  # Adjust if the sheet has a different name
    data = pd.read_excel(file_name, sheet_name=sheet_name)

    # Assuming the angles are in a column named 'Angles'
    grain_data = data['Major Diameter (µm)']

    # Calculate the average and standard deviation
    average_grain_size22 = np.mean(grain_data)
    std_dev22= np.std(grain_data)

    # Load the workbook and select the sheet
    wb = load_workbook(file_name)
    sheet = wb[sheet_name]

    # Write the results to the Excel file (e.g., in cells D1 and D2)
    sheet["H1"] = "Average Grain Size (µm)"
    sheet["I1"] = f"{average_grain_size22:.2f} "
    sheet["H2"] = "Standard Deviation (µm)"
    sheet["I2"] = f"{std_dev22:.2f} "
    sheet["H3"] = "Min diameter (µm)"
    sheet["I3"] = f"{min_diameter:.2f} "
    sheet["H4"] = "Max diameter (µm)"
    sheet["I4"] = f"{max_diameter:.2f} "

    # Save the updated workbook
    wb.save(file_name)

    

    ##########grain neck analysis #############################
    #import cv2
    #import numpy as np

    # Load the image
    #image = cv2.imread('2_enclosed_binary_seg.png')

    # Define color ranges
    #green = np.array([0, 255, 0])
    #red = np.array([0, 0, 255])
    #yellow = np.array([0, 255, 255])

    # Get the dimensions of the image
    #rows, cols, _ = image.shape

    # Create a copy of the image to modify
    #output_image = image.copy()

    # Define the directions for checking (horizontal, vertical, diagonal)
    #directions = [
        #(0, 1),  # Horizontal
        #(1, 0),  # Vertical
        #(1, 1),  # Diagonal down-right
       # (1, -1)  # Diagonal down-left
    #]

    # Iterate over the image
    #for row in range(rows):
        #for col in range(cols):
            # Check for the pattern in each direction
            #for dr, dc in directions:
                #for red_count in range(1, 4):  # Check for 1, 2, or 3 red pixels
                    #if (0 <= row + (red_count + 1) * dr < rows) and (0 <= col + (red_count + 1) * dc < cols):
                        #if (np.array_equal(image[row, col], green) and
                            #all(np.array_equal(image[row + i * dr, col + i * dc], red) for i in range(1, red_count + 1)) and
                            #np.array_equal(image[row + (red_count + 1) * dr, col + (red_count + 1) * dc], green)):

                            # Fill the red pixels with yellow and ensure continuity
                            #for i in range(1, red_count + 1):
                                #for j in range(-1, 2):  # To make the thickness 3 pixels
                                    #if 0 <= row + i * dr + j * dc < rows and 0 <= col + i * dc + j * dr < cols:
                                        #output_image[row + i * dr + j * dc, col + i * dc + j * dr] = yellow

    # Save and display the modified image

    #cv2.imshow('Modified Image', output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #cv2.imwrite("one_pixel_yel.png", output_image)







    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_enclosed_binary_seg.png")

    # Define color ranges
    green = np.array([0, 255, 0])
    red = np.array([0, 0, 255])
    yellow = np.array([0, 255, 255])

    # Get the dimensions of the image
    rows, cols, _ = image.shape

    # Create a copy of the image to modify
    output_image = image.copy()

    # Define the directions for checking (horizontal, vertical, diagonal)
    directions = [
        (0, 1),  # Horizontal
        (1, 0),  # Vertical
        (1, 1),  # Diagonal down-right
        (1, -1)  # Diagonal down-left
    ]

    # Iterate over the image
    for row in range(rows):
        for col in range(cols):
            # Check for the pattern in each direction
            for dr, dc in directions:
                for red_count in range(1, 4):  # Check for 1, 2, or 3 red pixels
                    if (0 <= row + (red_count + 1) * dr < rows) and (0 <= col + (red_count + 1) * dc < cols):
                        if (np.array_equal(image[row, col], green) and
                            all(np.array_equal(image[row + i * dr, col + i * dc], red) for i in range(1, red_count + 1)) and
                            np.array_equal(image[row + (red_count + 1) * dr, col + (red_count + 1) * dc], green)):

                            # Fill the red pixels with yellow and ensure continuity
                            for i in range(1, red_count + 1):
                                for j in range(-1, 2):  # To make the thickness 3 pixels
                                    if 0 <= row + i * dr + j * dc < rows and 0 <= col + i * dc + j * dr < cols:
                                        output_image[row + i * dr + j * dc, col + i * dc + j * dr] = yellow

    # Convert yellow to a binary mask
    yellow_mask = cv2.inRange(output_image, yellow, yellow)

    # Find contours of the yellow regions
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and remove thin yellow regions
    for contour in contours:
        if cv2.contourArea(contour) <= 3:  # Removing small yellow areas (1-pixel thick)
            cv2.drawContours(output_image, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Save and display the modified image
    #cv2.imshow('Modified Image', output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite(f"{z}_grains_neck_yellow_black.png", output_image)






    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_grains_neck_yellow_black.png")

    # Define the black color and the new green color
    black = np.array([0, 0, 0])
    green = np.array([0, 255, 0])

    # Replace black pixels with green pixels
    image[np.all(image == black, axis=-1)] = green

    # Save and display the modified image
   # cv2.imshow('Modified Image with Green Pixels', image)
   # cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite(f"{z}_grains_neck.png", image)






    import cv2
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.stats import norm

    # Load the image
    #image = cv2.imread(f"{z}_grains_neck_yellow_black.png")

    # Convert the image to HSV color space
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV (for detecting grains)
    #lower_green = np.array([40, 40, 40])
    #upper_green = np.array([70, 255, 255])

    # Define the range for yellow color in HSV (for detecting necks)
    #lower_yellow = np.array([20, 100, 100])
    #upper_yellow = np.array([40, 255, 255])

    # Threshold the image to isolate green (grain) and yellow (neck) regions
    #mask_green = cv2.inRange(hsv, lower_green, upper_green)
    #mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the green (grain) regions
    #contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get scale information
    #croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
    #croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
    #conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel

    # Initialize variables
    #valid_contours = 0
    #contour_data = []

    # Kernel for morphological dilation
    #kernel = np.ones((3, 3), np.uint8)

    # Process contours
    #for i, contour in enumerate(contours, 1):
        #if cv2.contourArea(contour) < 1:  # Skip small contours
            #continue

        # Create a mask for the current grain
        #grain_mask = np.zeros(mask_green.shape, dtype=np.uint8)
        #cv2.drawContours(grain_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Dilate the grain mask to find adjacent yellow pixels
        #dilated_grain_mask = cv2.dilate(grain_mask, kernel, iterations=1)
        #yellow_necks = cv2.bitwise_and(dilated_grain_mask, mask_yellow)

        # Count yellow pixels directly in contact with the grain
        #neck_pixel_count = cv2.countNonZero(yellow_necks)

        # Calculate the total length of necks in µm
        #total_neck_length = neck_pixel_count * conversion_factor

        # Calculate the perimeter of the grain in µm
        #perimeter = cv2.arcLength(contour, True) * conversion_factor

        # Calculate contiguity
        #contiguity = total_neck_length / perimeter if perimeter != 0 else 0

        # Update valid contours
        #valid_contours += 1
        #contour_data.append((i, total_neck_length, perimeter, contiguity))

        # Output details for each grain
        #print(f"Region {i}:")
        #print(f"  Total Neck Length: {total_neck_length:.2f} µm")
        #print(f"  Perimeter: {perimeter:.2f} µm")
        #print(f"  Contiguity: {contiguity:.4f}")
        #print()

    # Save output to Excel
    #df = pd.DataFrame(contour_data, columns=['Region', 'Total Neck Length (µm)', 'Perimeter (µm)', 'Contiguity'])
    #df.to_excel(f"{z}_enclosed_grain_neck_contiguty_details.xlsx", index=False)

    # Final output for all grains
    #if valid_contours == 0:
       # print("No valid contours found.")
    #else:
        #print("\nSummary of Contiguities:")
        #for data in contour_data:
            #region, neck_length, perimeter, contiguity = data
            #print(f"Region {region}: Total Neck Length = {neck_length:.2f} µm, Perimeter = {perimeter:.2f} µm, Contiguity = {contiguity:.4f}")
            
    
            
    
    import cv2
    import numpy as np

    # Step 1: Load the image
    image_path = f"{z}_grains_neck.png"
    image = cv2.imread(image_path)

    # Convert the image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Define color ranges for yellow and green
    # Adjust the ranges as necessary for your image
    yellow_lower = np.array([20, 100, 100])  # Lower bound for yellow in HSV
    yellow_upper = np.array([30, 255, 255])  # Upper bound for yellow in HSV

    green_lower = np.array([40, 50, 50])     # Lower bound for green in HSV
    green_upper = np.array([80, 255, 255])   # Upper bound for green in HSV

    # Create masks for yellow and green
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Step 3: Dilate the green mask to account for touching pixels
    kernel = np.ones((3, 3), np.uint8)  # Define a kernel size
    dilated_green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    # Find yellow pixels that are not touching green pixels
    isolated_yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(dilated_green_mask))

    # Step 4: Replace isolated yellow pixels with red
    # Create a red color array
    red_color = np.array([0, 0, 255], dtype=np.uint8)

    # Apply the mask to replace isolated yellow pixels with red
    image[isolated_yellow_mask > 0] = red_color

    # Step 5: Save the processed image
    output_path = f"{z}_grains_neck_1pix_yellow_neck.png"
    cv2.imwrite(output_path, image)

    print(f"Processed image saved to {output_path}")











    import cv2
    import numpy as np

    # Step 1: Load the image
    image_path = f"{z}_grains_neck_yellow_black.png"
    image = cv2.imread(image_path)

    # Convert the image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Define color ranges for yellow and green
    # Adjust the ranges as necessary for your image
    yellow_lower = np.array([20, 100, 100])  # Lower bound for yellow in HSV
    yellow_upper = np.array([30, 255, 255])  # Upper bound for yellow in HSV

    green_lower = np.array([40, 50, 50])     # Lower bound for green in HSV
    green_upper = np.array([80, 255, 255])   # Upper bound for green in HSV

    # Create masks for yellow and green
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Step 3: Dilate the green mask to account for touching pixels
    kernel = np.ones((3, 3), np.uint8)  # Define a kernel size
    dilated_green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    # Find yellow pixels that are not touching green pixels
    isolated_yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(dilated_green_mask))

    # Step 4: Replace isolated yellow pixels with red
    # Create a red color array
    red_color = np.array([0, 0, 255], dtype=np.uint8)

    # Apply the mask to replace isolated yellow pixels with red
    image[isolated_yellow_mask > 0] = red_color

    # Step 5: Save the processed image
    output_path = f"{z}_grains_neck_yellow_black_1pix_yellow_neck.png"
    cv2.imwrite(output_path, image)

    print(f"Processed image saved to {output_path}")


















    import cv2
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    # Load the image
    image = cv2.imread(f"{z}_grains_neck_yellow_black_1pix_yellow_neck.png")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV (for detecting grains)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Define the range for yellow color in HSV (for detecting necks)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Threshold the image to isolate green (grain) and yellow (neck) regions
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the green (grain) regions
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get scale information
    #croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
    #croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
    #conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel

    # Initialize variables
    valid_contours = 0
    contour_data = []

    # Kernel for morphological dilation
    kernel = np.ones((3, 3), np.uint8)

    # Process contours and store data
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < 1:  # Skip small contours
            continue

        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Create a mask for the current grain
        grain_mask = np.zeros(mask_green.shape, dtype=np.uint8)
        cv2.drawContours(grain_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Dilate the grain mask to find adjacent yellow pixels
        dilated_grain_mask = cv2.dilate(grain_mask, kernel, iterations=1)
        yellow_necks = cv2.bitwise_and(dilated_grain_mask, mask_yellow)

        # Count yellow pixels directly in contact with the grain
        neck_pixel_count = cv2.countNonZero(yellow_necks)

        # Calculate the total length of necks in µm
        total_neck_length = neck_pixel_count * conversion_factor

        # Calculate the perimeter of the grain in µm
        perimeter = cv2.arcLength(contour, True) * conversion_factor

        # Calculate contiguity
        contiguity = total_neck_length / perimeter if perimeter != 0 else 0

        # Update valid contours
        valid_contours += 1
        contour_data.append((contour, cX, cY, valid_contours, total_neck_length, perimeter, contiguity))

    # Prepare the output image for displaying the labels
    output_image = image.copy()

    # Draw contours and label numbers
    for contour, cX, cY, region_number, total_neck_length, perimeter, contiguity in contour_data:
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        text = str(region_number)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        textX = cX - text_size[0] // 2
        textY = cY + text_size[1] // 2
        cv2.putText(output_image, text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the labeled image
    cv2.imwrite(f"{z}_grain_neck_regions1.png", output_image)
    #cv2.imshow("Labeled Image", output_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save output to Excel
    df = pd.DataFrame(contour_data, columns=['Contour', 'Centroid X', 'Centroid Y', 'Region', 'Total Neck Length (µm)', 'Perimeter (µm)', 'Contiguity'])
    df.drop(columns=['Contour'], inplace=True)  # Drop the contour column for cleaner Excel output
    df.to_excel(f"{z}_enclosed_grain_neck_contiguity_details.xlsx", index=False)

    # Final output for all grains
    if valid_contours == 0:
        print("No valid contours found.")
    else:
        print("\nSummary of Contiguities:")
        for data in contour_data:
            region, neck_length, perimeter, contiguity = data[3], data[4], data[5], data[6]
            print(f"Region {region}: Total Neck Length = {neck_length:.2f} µm, Perimeter = {perimeter:.2f} µm, Contiguity = {contiguity:.4f}")

            
      

    import pandas as pd
    import numpy as np
    from openpyxl import load_workbook

    # Load the Excel file
    file_name = f"{z}_enclosed_grain_neck_contiguity_details.xlsx"
    sheet_name = "Sheet1"  # Adjust if the sheet has a different name
    data = pd.read_excel(file_name, sheet_name=sheet_name)

    # Assuming the angles are in a column named 'Angles'
    contiguity = data['Contiguity']

    # Calculate the average and standard deviation
    average_contiguity11_ = np.mean(contiguity)
    std_dev11_ = np.std(contiguity)

    # Load the workbook and select the sheet
    wb = load_workbook(file_name)
    sheet = wb[sheet_name]

    # Write the results to the Excel file (e.g., in cells D1 and D2)
    sheet["H1"] = "Average Contiguity"
    sheet["I1"] = f"{average_contiguity11_:.2f} "
    sheet["H2"] = "Standard Deviation"
    sheet["I2"] = f"{std_dev11_:.2f} "

    # Save the updated workbook
    wb.save(file_name)



    ########extract yellow pixels



    import cv2
    import numpy as np

    # Step 1: Read the image
    image = cv2.imread(f"{z}_grains_neck.png")

    # Step 2: Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 3: Define the HSV range for yellow
    # Note: You may need to adjust these values based on your image
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Step 4: Create a mask for yellow pixels
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Step 5: Convert yellow pixels to white and background to black
    result_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    result_image[yellow_mask > 0] = [255, 255, 255]

    # Step 6: Convert the result to grayscale
    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Step 7: Save or display the result
    #cv2.imshow('Result', gray_image)
    cv2.imwrite(f"{z}_grains_neck_ex.png", gray_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()






    #######thin img


    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage import img_as_ubyte

    # Load the image in grayscale mode
    image = cv2.imread(f"{z}_grains_neck_ex.png", cv2.IMREAD_GRAYSCALE)

    # Ensure the white boundary is represented as 255 and the background as 0
    # If your white boundary is not pure white (255), you may need to threshold it
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Perform morphological erosion to thin the boundary
    # The kernel defines the neighborhood size for erosion, adjust if necessary
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=0)

    # Skeletonize the image to ensure the boundary is thinned
    skeleton = skeletonize(eroded_image // 255)  # Skeletonize expects a binary image (0, 1)
    skeleton_image = img_as_ubyte(skeleton)  # Convert back to 0-255 range

    # Dilate the skeleton to ensure uniform 2-pixel thickness boundary
    thinned_image = cv2.dilate(skeleton_image, kernel, iterations=0)

    # Save the output image

    # Display the result (optional)

    cv2.imwrite(f"{z}_grains_neck_ex_thin.png", thinned_image)
    #cv2.imshow("image",thinned_image)
    ##cv2.waitKey(0)
    #cv2.destroyAllWindows()






    #######filling yellow pixels with green

    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(f"{z}_grains_neck.png")

    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for yellow color
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype=np.uint8)

    # Create a mask for yellow pixels
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Define the green color in BGR
    green_color = [0, 255, 0]

    # Replace yellow pixels with green
    image[yellow_mask > 0] = green_color

    # Save the modified image
    cv2.imwrite(f"{z}_grains_neck_yellow_to_green.png", image)

    # Display the original and modified images (optional)

    #cv2.imshow('Modified Image', image)
    ##cv2.waitKey(0)
    #cv2.destroyAllWindows()




    import cv2
    import numpy as np
    from skimage.morphology import disk, dilation
    from skimage import io, img_as_ubyte
    from skimage.color import rgb2gray
    from skimage.feature import canny

    # Read the original image
    original_image = cv2.imread(f"{z}_grains_neck_yellow_to_green.png")

    # Convert to grayscale
    gray_image = rgb2gray(original_image)

    # Detect edges using the Canny method
    edge_image = canny(gray_image)

    # Define a structuring element for dilation (disk of radius 0)
    selem = disk(0)

    # Dilate the edges
    dilated_edges = dilation(edge_image, selem)

    # Convert the dilated edge image to uint8 type for proper overlay
    dilated_edges_uint8 = img_as_ubyte(dilated_edges)

    # Overlay the dilated edges on the original image (in white)
    overlayed_image = original_image.copy()
    overlayed_image[dilated_edges_uint8 > 0] = [255, 255, 255]

    # Display the dilated edges


    # Save the result image
    cv2.imwrite(f"{z}_grains_neck_yellow_to_green_perimeter.png", overlayed_image)





    ####merge img contiguity

    import cv2
    import numpy as np
    # Load the original image and the boundary image
    originalImg = cv2.imread(f"{z}_grains_neck_yellow_to_green_perimeter.png")
    boundaryImg = cv2.imread(f"{z}_grains_neck_ex_thin.png")

    print(originalImg.shape,boundaryImg.shape)
    # Ensure the images are of the same size
    assert originalImg.shape == boundaryImg.shape, 'Images must be the same size'
    # Convert boundary image to binary (white boundary on black background)
    boundaryMask = boundaryImg[:, :, 0] > 100
    # Create an output image initialized with the original image
    outputImg = originalImg.copy()
    # Set the boundary pixels to white
    outputImg[boundaryMask] = [255, 255, 255]
    # Display the merged image


    cv2.imwrite(f"{z}_grains_neck_merged.png", outputImg)






    ###########contiguity


   #num_rows=int(input("Enter number of lines for enclosed grain contiguity : "))
   #import cv2
   #import numpy as np
   #import pandas as pd  # For saving data to Excel
   #import matplotlib.pyplot as plt

   #def draw_line_and_mark_intersections(image, row_index):
       # Define the colors
       #tungsten_color = np.array([0, 255, 0])  # green
       #boundary_color = np.array([255, 255, 255])  # white
       #matrix_color = np.array([255, 0, 0])  # red
       
       # Extract the row from the image
       #row = image[row_index, :, :]
       
       # Initialize lists to store intersection points
       #intersections = []
       
       # Process the row to find intersections with white pixels
       #for i in range(row.shape[0]):
           #pixel = row[i]
           #if np.array_equal(pixel, boundary_color):
               #intersections.append((row_index, i))  # Mark this point for intersection
       
       # Initialize interaction counters
       #tungsten_tungsten_interactions = 0
       ##tungsten_matrix_interactions = 0
       #matrix_tungsten_interactions = 0
       
       # Process intersections to count interactions
       #for point in intersections:
           #y, x = point
           
           # Check left and right pixels
           #left_pixel = row[x - 1] if x - 1 >= 0 else None
           #right_pixel = row[x + 1] if x + 1 < row.shape[0] else None
           
           #if left_pixel is not None and right_pixel is not None:
               #if np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, tungsten_color):
                   #tungsten_tungsten_interactions += 1
               #elif np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, matrix_color):
                   #tungsten_matrix_interactions += 1
               #elif np.array_equal(left_pixel, matrix_color) and np.array_equal(right_pixel, tungsten_color):
                   #matrix_tungsten_interactions += 1

       # Check for cases with consecutive boundary pixels
       #for i in range(len(intersections) - 1):
           #curr_point = intersections[i]
           #next_point = intersections[i + 1]

           # Left pixel of current point and right pixel of next point
           #left_pixel = row[curr_point[1] - 1] if curr_point[1] - 1 >= 0 else None
           #right_pixel = row[next_point[1] + 1] if next_point[1] + 1 < row.shape[0] else None
           
           #if left_pixel is not None and right_pixel is not None:
               #if np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, matrix_color):
                   #tungsten_matrix_interactions += 1
               #elif np.array_equal(left_pixel, matrix_color) and np.array_equal(right_pixel, tungsten_color):
                   #matrix_tungsten_interactions += 1

       # Calculate contiguity
       #denominator = (2 * tungsten_tungsten_interactions + tungsten_matrix_interactions + matrix_tungsten_interactions)
       #if denominator != 0:
       #    contiguity = (2 * tungsten_tungsten_interactions) / denominator
       #else:
           #contiguity = 0
       
       #return contiguity

   #def calculate_average_contiguity(image_path, num_rows, output_excel=f"{z}_enclosed_grain_linear_contiguity_details.xlsx"):
       # Load the image
       #image = cv2.imread(image_path)
       #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
       # Determine the factor
       #factor = image_rgb.shape[0] / num_rows
       
       # Create a DataFrame to store results
       ##contiguity_data = []

       # Calculate contiguity for each row and take the average
       #total_contiguity = 0
       #valid_rows = 0  # Count rows with contiguity > 0.1
       #for i in range(num_rows):
           ##row_index = int(i * factor)
          # contiguity = draw_line_and_mark_intersections(image_rgb, row_index)
           #i#f contiguity >= 0:  # Filter for contiguity > 0.1
              # total_contiguity += contiguity
              ## valid_rows += 1
               #print(f"Row {row_index}: Contiguity = {contiguity}")
               ## Append row index and contiguity to the data list
               #contiguity_data.append({"Row Index": row_index, "Contiguity": contiguity})
       
       # Calculate average contiguity
       #if valid_rows > 0:
           #average_contiguity = total_contiguity / valid_rows
      # else:
          # average_contiguity = 0  # Handle case where no rows meet the condition
       
       # Save the data to Excel
      ## df = pd.DataFrame(contiguity_data)
       #df.to_excel(output_excel, index=False)
       
       # Calculate standard deviation
       #std_dev1= np.std(df['Contiguity'])
       
       #return average_contiguity, std_dev1


   # Example usage
   ##image_path = f"{z}_grains_neck_merged.png"  # Change this to the path of your image
   #average_contiguity, std_dev1 = calculate_average_contiguity(image_path, num_rows, output_excel=f"{z}_enclosed_grain_linear_contiguity_details.xlsx")



   #import pandas as pd
   ##import numpy as np
   #from openpyxl import load_workbook

   # Load the Excel file
   #file_name = f"{z}_enclosed_grain_linear_contiguity_details.xlsx"
   #sheet_name = "Sheet1"  # Adjust if the sheet has a different name
   #data = pd.read_excel(file_name, sheet_name=sheet_name)

   # Assuming the angles are in a column named 'Angles'
  # contiguity = data['Contiguity']

   # Calculate the average and standard deviation
   #average_contiguity11 = np.mean(contiguity)
   ##std_dev11 = np.std(contiguity)

   # Load the workbook and select the sheet
   #wb = load_workbook(file_name)
   #sheet = wb[sheet_name]

   # Write the results to the Excel file (e.g., in cells D1 and D2)
   #sheet["D1"] = "Average Contiguity"
   ##sheet["E1"] = f"{average_contiguity11:.2f} "
   #sheet["D2"] = "Standard Deviation"
   #sheet["E2"] = f"{std_dev11:.2f} "

   # Save the updated workbook
   #wb.save(file_name)









    num_rows=int(input("Enter number of lines to be used for contiguity measurement : "))

    import cv2
    import numpy as np
    import pandas as pd  # For saving data to Excel
    import matplotlib.pyplot as plt

    def draw_line_and_mark_intersections(image, row_index):
        # Define the colors
        tungsten_color = np.array([0, 255, 0])  # green
        boundary_color = np.array([255, 255, 255])  # white
        matrix_color = np.array([255, 0, 0])  # red
        
        # Extract the row from the image
        row = image[row_index, :, :]
        
        # Initialize lists to store intersection points
        intersections = []
        
        # Process the row to find intersections with white pixels
        for i in range(row.shape[0]):
            pixel = row[i]
            if np.array_equal(pixel, boundary_color):
                intersections.append((row_index, i))  # Mark this point for intersection
        
        # Initialize interaction counters
        tungsten_tungsten_interactions = 0
        tungsten_matrix_interactions = 0
        matrix_tungsten_interactions = 0
        
        # Process intersections to count interactions
        for point in intersections:
            y, x = point
            
            # Check left and right pixels
            left_pixel = row[x - 1] if x - 1 >= 0 else None
            right_pixel = row[x + 1] if x + 1 < row.shape[0] else None
            
            if left_pixel is not None and right_pixel is not None:
                if np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, tungsten_color):
                    tungsten_tungsten_interactions += 1
                elif np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, matrix_color):
                    tungsten_matrix_interactions += 1
                elif np.array_equal(left_pixel, matrix_color) and np.array_equal(right_pixel, tungsten_color):
                    matrix_tungsten_interactions += 1

        # Check for cases with consecutive boundary pixels
        for i in range(len(intersections) - 1):
            curr_point = intersections[i]
            next_point = intersections[i + 1]

            # Left pixel of current point and right pixel of next point
            left_pixel = row[curr_point[1] - 1] if curr_point[1] - 1 >= 0 else None
            right_pixel = row[next_point[1] + 1] if next_point[1] + 1 < row.shape[0] else None
            
            if left_pixel is not None and right_pixel is not None:
                if np.array_equal(left_pixel, tungsten_color) and np.array_equal(right_pixel, matrix_color):
                    tungsten_matrix_interactions += 1
                elif np.array_equal(left_pixel, matrix_color) and np.array_equal(right_pixel, tungsten_color):
                    matrix_tungsten_interactions += 1

        # Calculate contiguity
        denominator = (2 * tungsten_tungsten_interactions + tungsten_matrix_interactions + matrix_tungsten_interactions)
        if denominator != 0:
            contiguity = (2 * tungsten_tungsten_interactions) / denominator
        else:
            contiguity = 0
        
        return contiguity

    def calculate_average_contiguity(image_path, num_rows, output_excel=f"{z}_contiguity_results.xlsx"):
        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Determine the factor
        factor = image_rgb.shape[0] / num_rows
        
        # Create a DataFrame to store results
        contiguity_data = []

        # Calculate contiguity for each row and take the average
        total_contiguity = 0
        valid_rows = 0  # Count rows with contiguity > 0.1
        for i in range(num_rows):
            row_index = int(i * factor)
            contiguity = draw_line_and_mark_intersections(image_rgb, row_index)
            if contiguity >= 0:  # Filter for contiguity > 0.1
                total_contiguity += contiguity
                valid_rows += 1
                print(f"Row {row_index}: Contiguity = {contiguity}")
                # Append row index and contiguity to the data list
                contiguity_data.append({"Row Index": row_index, "Contiguity": contiguity})
        
        # Calculate average contiguity
        if valid_rows > 0:
            average_contiguity = total_contiguity / valid_rows
        else:
            average_contiguity = 0  # Handle case where no rows meet the condition
        
        # Save the data to Excel
        df = pd.DataFrame(contiguity_data)
        df.to_excel(output_excel, index=False)
        
        # Calculate standard deviation
        std_dev1= np.std(df['Contiguity'])
        
        return average_contiguity, std_dev1


    # Example usage
    image_path = f"{z}_merged_thin_img.png"  # Change this to the path of your image
    average_contiguity, std_dev1 = calculate_average_contiguity(image_path, num_rows, output_excel=f"{z}_contiguity_results.xlsx")



    import pandas as pd
    import numpy as np
    from openpyxl import load_workbook

    # Load the Excel file
    file_name = f"{z}_contiguity_results.xlsx"
    sheet_name = "Sheet1"  # Adjust if the sheet has a different name
    data = pd.read_excel(file_name, sheet_name=sheet_name)

    # Assuming the angles are in a column named 'Angles'
    contiguity = data['Contiguity']

    # Calculate the average and standard deviation
    average_contiguity11 = np.mean(contiguity)
    std_dev11 = np.std(contiguity)

    # Load the workbook and select the sheet
    wb = load_workbook(file_name)
    sheet = wb[sheet_name]

    # Write the results to the Excel file (e.g., in cells D1 and D2)
    sheet["D1"] = "Average Contiguity"
    sheet["E1"] = f"{average_contiguity11:.2f} "
    sheet["D2"] = "Standard Deviation"
    sheet["E2"] = f"{std_dev11:.2f} "

    # Save the updated workbook
    wb.save(file_name)
    
    
    
    
    
    image = cv2.imread(f"{z}.png")

    # Convert the image to grayscale for thresholding
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the thresholded image using matplotlib
    plt.imshow(thresh, cmap='gray')
    plt.show()

    #cv2.imshow("threshold_image",thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save the thresholded image
    plt.imsave(f"{z}_thresh_precipitate.png", thresh, cmap='gray')


    img=cv2.imread(f"{z}_thresh_precipitate.png",0)
    color1=img==0
    color2=img==255
    allsegment=np.zeros((img.shape[0],img.shape[1],3))
    allsegment[color1]=(0,0,0)
    allsegment[color2]=(0,1,0)
    plt.imsave(f"{z}_thresh_precipitate_green.png",allsegment)




    import cv2
    import numpy as np

    def filter_green_clusters(image_path, min_pixels, max_pixels):
        # Step 1: Load the image in color
        image = cv2.imread(image_path)
        
        # Step 2: Create a mask for green pixels (threshold for green)
        # Define the lower and upper bounds for the green color in BGR format
        lower_green = np.array([0, 100, 0])  # Lower bound for green (BGR)
        upper_green = np.array([100, 255, 100])  # Upper bound for green (BGR)

        # Threshold the image to get only green pixels
        green_mask = cv2.inRange(image, lower_green, upper_green)

        # Step 3: Find connected components for green pixels
        num_labels, labels_im = cv2.connectedComponents(green_mask)

        # Create an output image, initially all black
        output_image = np.zeros_like(green_mask)

        # Step 4: Filter the connected components based on the pixel count range
        for label in range(1, num_labels):  # Start from 1 because 0 is the background
            # Get all the pixels that belong to this component
            component = (labels_im == label).astype(np.uint8)

            # Count the number of green pixels in this component
            pixel_size = np.sum(component)

            # Step 5: If the component size is within the specified range, retain it
            if min_pixels <= pixel_size <= max_pixels:
                output_image[component == 1] = 255  # Set pixels of this component to white

        # Step 6: Save and display the output image
        cv2.imwrite(f"{z}_thresh_precipitate_particles.png", output_image)
        #cv2.imshow('Filtered Green Clusters', output_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    # Usage example
    image_path = f"{z}_thresh_precipitate_green.png"  # Path to the input image
    filter_green_clusters(image_path, 0, max_precipitate_size_pixel)






    img=cv2.imread(f"{z}_thresh_precipitate_particles.png",0)
    seg111=img==0
    seg222=img==255
    allsegment=np.zeros((img.shape[0],img.shape[1],3))
    allsegment[seg111]=(1,0,0)
    allsegment[seg222]=(0,1,0)

    if (np.sum(seg222==1)*100)/(np.sum(seg222==1)+np.sum(seg222==0))==0:
        import os

        # Specify the path to the file
        file_path = f"{z}_thresh_precipitate_particles.png"

        # Check if the file exists before attempting to delete
        if os.path.exists(file_path):
           os.remove(file_path)    
        print("There are no Tungsten precipitate in the Micrograph")
    else:
        print("The are Tungsten precipitates")
        
        img=cv2.imread(f"{z}_thresh_precipitate_particles.png",0)
        seg_1_1=img==0
        seg_2_2=img==255
        allsegment=np.zeros((img.shape[0],img.shape[1],3))
        allsegment[seg_1_1]=(1,0,0)
        allsegment[seg_2_2]=(0,1,0)
        plt.imsave(f"{z}_thresh_precipitate_particles_seg.png",allsegment)





        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        # Load the image
        image = cv2.imread(f"{z}_thresh_precipitate_particles_seg.png")

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])

        # Threshold the image to isolate green regions
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours of the green regions
        contours_, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image for drawing
        output_image = image.copy()

        # Get scale information
        #croped_img_scale_um = float(input("Enter original image scale bar length in µm: "))
        #croped_img_scale_length = float(input("Enter original image scale bar length in pixels: "))
        #conversion_factor = croped_img_scale_um / croped_img_scale_length  # µm/pixel

        # Initialize variables
        valid_contours_ = 0
        total_major_diameter_ = 0
        contour_data_ = []
        major_diameters_ = []

        # Process contours
        for i, contour_ in enumerate(contours_, 1):
            if cv2.contourArea(contour_) < 0:  # Skip small contours
                continue

            moments_ = cv2.moments(contour_)
            if moments_['m00'] != 0:
                cX = int(moments_['m10'] / moments_['m00'])
                cY = int(moments_['m01'] / moments_['m00'])

                # Bounding box and major diameter
                x, y, w, h = cv2.boundingRect(contour_)
                major_diameter_ = max(w, h) * conversion_factor  # µm
                area_um2_ = cv2.contourArea(contour_) * conversion_factor ** 2
                perimeter_um_ = cv2.arcLength(contour_, True) * conversion_factor
                particle_edge_um_ = area_um2_ ** 0.5
                circularity_um_ = 4 * 3.14 * area_um2_ / (perimeter_um_ ** 2)

                # Update variables
                total_major_diameter_ += major_diameter_
                major_diameters_.append(major_diameter_)
                contour_data_.append((contour_, cX, cY, major_diameter_))
                valid_contours_ += 1

                # Output details
                print(f'Region {valid_contours_}:')
                print(f'  Area: {area_um2_:.2f} µm²')
                print(f'  Perimeter: {perimeter_um_:.2f} µm')
                print(f'  Major Diameter: {major_diameter_:.2f} µm')
                print(f'  Particle Edge: {particle_edge_um_:.2f} µm')
                print(f'  Circularity: {circularity_um_:.2f} µm')
                print()

        # Final calculations
        if valid_contours_ > 0:
            average_major_diameter_ = total_major_diameter_ / valid_contours_
            min_diameter_ = min(major_diameters_)
            max_diameter_ = max(major_diameters_)
            # Calculate the standard deviation using numpy
            std_dev_ = np.std(major_diameters_)
            print(f"\nAverage Major Diameter: {average_major_diameter_:.2f} µm")
            print(f"Minimum Major Diameter: {min(major_diameters_):.2f} µm")
            print(f"Maximum Major Diameter: {max(major_diameters_):.2f} µm")
            print(f"Standard Deviation: {np.std(major_diameters_):.2f} µm")

        # Draw contours and label numbers
        for i, (contour_, cX, cY, major_diameter_) in enumerate(contour_data_, 1):
            cv2.drawContours(output_image, [contour_], -1, (0, 255, 0), 2)
            text_ = str(i)
            text_size_ = cv2.getTextSize(text_, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            textX_ = cX - text_size_[0] // 2
            textY_ = cY + text_size_[1] // 2
            cv2.putText(output_image, text_, (textX_, textY_), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save and display results
        cv2.imwrite(f"{z}_thresh_precipitate_particles_regions.png", output_image)
        #cv2.imshow("Labeled Image", output_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Particle size distribution
        if major_diameters_:
            major_diameters_ = np.array(major_diameters_)
            hist, bins = np.histogram(major_diameters_, bins='auto')
            bin_centers_ = 0.5 * (bins[1:] + bins[:-1])
            mu, std = norm.fit(major_diameters_)

            plt.bar(bin_centers_, hist, width=(bins[1] - bins[0]), alpha=0.6, color='purple', edgecolor='black', linewidth=1.0)
            x = np.linspace(bin_centers_[0], bin_centers_[-1], 100)
            p = norm.pdf(x, mu, std) * len(major_diameters_) * (bins[1] - bins[0])
            plt.plot(x, p, 'b', linewidth=2)
            plt.title('W Precipitates Size Distribution')
            plt.xlabel('W Precipitates Size (µm)')
            plt.ylabel('Count')
            plt.xlim([0, max(major_diameters_) + 10])
            plt.ylim([0, max(hist) + 5])
            plt.savefig(f"{z}_thresh_precipitate_particles_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            
            
            
            
        import pandas as pd

        # Initialize a list to store the details of each valid region
        region_details = []
        valid_contours = 0  # Counter for valid regions

        # Process each contour and store the details
        for contour_ in contours_:
            if cv2.contourArea(contour_) < 0:  # Ignore small contours
                continue

            moments = cv2.moments(contour_)
            if moments['m00'] != 0:
                x, y, w, h = cv2.boundingRect(contour_)
                major_diameter_ = max(w, h) * conversion_factor  # Major diameter in µm
                area_um2_ = cv2.contourArea(contour_) * conversion_factor ** 2
                perimeter_um_ = cv2.arcLength(contour_, True) * conversion_factor
                particle_edge_um_ = area_um2_ ** 0.5  # Corrected to compute the square root of area_um2
                circularity_um_ = 4 * 3.14 * area_um2_ / (perimeter_um_ ** 2)  # Corrected to square the perimeter_um variable

                # Increment valid region counter
                valid_contours += 1

                # Append details to the list
                region_details.append({
                    'Region': valid_contours,  # Use valid_contours as region index
                    'Area (µm²)': round(area_um2_, 2),
                    'Perimeter (µm)': round(perimeter_um_, 2),
                    'Major Diameter (µm)': round(major_diameter_, 2),
                    'Particle Edge (µm)': round(particle_edge_um_, 2),
                    'Circularity (µm)': round(circularity_um_, 2)
                })

        # Ensure all valid regions match the image labels
        if len(region_details) != valid_contours:
            raise ValueError("Mismatch between the number of labeled regions and saved region details.")

        # Create a pandas DataFrame from the region details
        df = pd.DataFrame(region_details)

        # Save the DataFrame to an Excel file
        output_file = f"{z}_precipitates_region_details.xlsx"
        df.to_excel(output_file, index=False)

        print(f"Region details successfully saved to {output_file}")




        import pandas as pd
        import numpy as np
        from openpyxl import load_workbook

        # Load the Excel file
        file_name = f"{z}_precipitates_region_details.xlsx"
        sheet_name = "Sheet1"  # Adjust if the sheet has a different name
        data = pd.read_excel(file_name, sheet_name=sheet_name)

        # Assuming the angles are in a column named 'Angles'
        grain_data = data['Major Diameter (µm)']

        # Calculate the average and standard deviation
        average_grain_size22_ = np.mean(grain_data)
        std_dev22_= np.std(grain_data)

        # Load the workbook and select the sheet
        wb = load_workbook(file_name)
        sheet = wb[sheet_name]

        # Write the results to the Excel file (e.g., in cells D1 and D2)
        sheet["H1"] = "Average Precipitate Size (µm)"
        sheet["I1"] = f"{average_grain_size22_:.2f} "
        sheet["H2"] = "Standard Deviation (µm)"
        sheet["I2"] = f"{std_dev22_:.2f} "
        sheet["H3"] = "Min diameter (µm)"
        sheet["I3"] = f"{min_diameter_:.2f} "
        sheet["H4"] = "Max diameter (µm)"
        sheet["I4"] = f"{max_diameter_:.2f} "

        # Save the updated workbook
        wb.save(file_name)

    
    
    
    
    
    ############################################################DI-HYDRAL ANGLE
    
    
    ############################Finding boundary perimeter of a segmented image or threshold image
    import cv2
    import numpy as np
    from skimage.morphology import disk, dilation
    from skimage import io, img_as_ubyte
    from skimage.color import rgb2gray
    from skimage.feature import canny
    
    # Read the original image
    original_image = cv2.imread(f"{z}_thresh_cleaned_seg.png")
    
    # Convert to grayscale
    gray_image = rgb2gray(original_image)
    
    # Detect edges using the Canny method
    edge_image = canny(gray_image)
    
    # Define a structuring element for dilation (disk of radius 0)
    selem = disk(0)
    
    # Dilate the edges
    dilated_edges = dilation(edge_image, selem)
    
    # Convert the dilated edge image to uint8 type for proper overlay
    dilated_edges_uint8 = img_as_ubyte(dilated_edges)
    
    # Overlay the dilated edges on the original image (in white)
    overlayed_image = original_image.copy()
    overlayed_image[dilated_edges_uint8 > 0] = [255, 255, 255]
    
    # Display the dilated edges
    
    
    # Save the result image
    cv2.imwrite(f"{z}_perimeter.png", overlayed_image)
    
    
    
    
    
    
    
    ##################################### 1st condition
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter.png")
    
    # Define color ranges
    white_pixel = [255, 255, 255]  # White boundary pixel
    green_pixel = [0, 255, 0]  # Green pixel
    blue_pixel = [255, 0, 0]  # Blue pixel (BGR format)
    
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Check conditions and fill the white boundary pixel with blue
    def is_green(pixel):
        return np.array_equal(pixel, green_pixel)
    
    def is_white(pixel):
        return np.array_equal(pixel, white_pixel)
    
    # Loop through each pixel to find white boundary pixels and check conditions
    for y in range(1, height-1):  # Avoiding the edges
        for x in range(1, width-1):  # Avoiding the edges
            if is_white(image[y, x]):
                # Check horizontal direction (left and right)
                left_green = all(is_green(image[y, x-i]) for i in range(3, 8) if x-i >= 0) and is_white(image[y, x-1])
                right_green = all(is_green(image[y, x+i]) for i in range(3, 8) if x+i < width) and is_white(image[y, x+1])
                
                # Check vertical direction (up and down)
                up_green = all(is_green(image[y-i, x]) for i in range(3, 8) if y-i >= 0) and is_white(image[y-1, x])
                down_green = all(is_green(image[y+i, x]) for i in range(3, 8) if y+i < height) and is_white(image[y+1, x])
                
                # If either condition is met, fill the white pixel with blue
                if (left_green and right_green) or (up_green and down_green):
                    image[y, x] = blue_pixel
    
    # Save the modified image
    cv2.imwrite(f"{z}_perimeter_1st.png", image)
    
    
    
    
    
    
    
    ###################### 2nd condition
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter.png")
    
    # Convert to RGB (OpenCV loads images in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the pixel values for colors
    WHITE = [255, 255, 255]  # White boundary pixels
    GREEN = [0, 255, 0]      # Green grain pixels
    BLUE = [0, 0, 255]       # Blue for filling the selected boundary pixels
    
    # Function to check the conditions
    def check_conditions(image, x, y):
        # Ensure we are not on the edge of the image
        if x - 7 >= 0 and y - 7 >= 0 and x + 7 < image.shape[1] and y + 7 < image.shape[0]:
            # Extract the pixel neighbors #3 best white pixels
            right_4 = image[y, x+1:x+4]
            left_7 = image[y, x-7:x]
            down_4 = image[y+1:y+4, x]
            up_7 = image[y-7:y, x]
            
            right_7 = image[y, x+1:x+8]
            left_4 = image[y, x-3:x]
            down_7 = image[y+1:y+8, x]
            up_4 = image[y-3:y, x]
            
            # Check the four conditions
            if (np.all(right_4 == WHITE) and np.all(left_7 == GREEN) and np.all(down_4 == WHITE) and np.all(up_7 == GREEN)):
                return True
            if (np.all(right_7 == GREEN) and np.all(left_4 == WHITE) and np.all(down_4 == WHITE) and np.all(up_7 == GREEN)):
                return True
            if (np.all(right_7 == GREEN) and np.all(left_4 == WHITE) and np.all(down_7 == GREEN) and np.all(up_4 == WHITE)):
                return True
            if (np.all(right_4 == WHITE) and np.all(left_7 == GREEN) and np.all(down_7 == GREEN) and np.all(up_4 == WHITE)):
                return True
        return False
    
    # Loop over the image to find individual white pixels
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Check if the current pixel is white
            if np.all(image_rgb[y, x] == WHITE):
                # Check if the conditions are met
                if check_conditions(image_rgb, x, y):
                    # Fill with blue if conditions are met
                    image_rgb[y, x] = BLUE
    
    # Convert the image back to BGR and save
    result_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{z}_perimeter_2nd.png", result_image)
    
    
    
    
    
    
    
    
    ###Extraction of blue and white
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load the image
    image_path = f"{z}_perimeter_1st.png"
    image = cv2.imread(image_path)
    
    # Convert the image to HSV (Hue, Saturation, Value) for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for white and blue pixels in HSV
    # White color range
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 55, 255], dtype=np.uint8)
    
    # Blue color range
    lower_blue = np.array([100, 150, 0], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    
    # Create masks for white and blue colors
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Combine the two masks
    combined_mask = cv2.bitwise_or(white_mask, blue_mask)
    
    # Extract only the white and blue pixels, setting the rest to black
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    
    # Save the result image
    cv2.imwrite(f"{z}_perimeter_1st_ex.png", result)
    
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load the image
    image_path = f"{z}_perimeter_2nd.png"
    image = cv2.imread(image_path)
    
    # Convert the image to HSV (Hue, Saturation, Value) for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for white and blue pixels in HSV
    # White color range
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 55, 255], dtype=np.uint8)
    
    # Blue color range
    lower_blue = np.array([100, 150, 0], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    
    # Create masks for white and blue colors
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Combine the two masks
    combined_mask = cv2.bitwise_or(white_mask, blue_mask)
    
    # Extract only the white and blue pixels, setting the rest to black
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    
    # Save the result image
    cv2.imwrite(f"{z}_perimeter_2nd_ex.png", result)
    
    
    
    
    ################## adjusting blue pixels
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter_1st_ex.png")
    
    # Threshold the image to isolate white grain boundary and blue pixels
    # Blue pixel range (B, G, R): (255, 0, 0)
    blue_pixels = np.where((image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))
    
    # Create a copy of the image to modify
    output_image = image.copy()
    
    # Define function to calculate Euclidean distance between two points
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    # Process each blue pixel
    for i in range(len(blue_pixels[0])):
        x1, y1 = blue_pixels[0][i], blue_pixels[1][i]
        
        for j in range(i + 1, len(blue_pixels[0])):
            x2, y2 = blue_pixels[0][j], blue_pixels[1][j]
            
            # Check if the two blue pixels are between 2-5 pixels apart
            distance = euclidean_distance((x1, y1), (x2, y2))
            if 1 <= distance <= 5:
                # Calculate the midpoint
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Remove the original blue pixels
                output_image[x1, y1] = [255, 255, 255]  # Set to white (original boundary color)
                output_image[x2, y2] = [255, 255, 255]  # Set to white (original boundary color)
                
                # Place the resultant blue pixel at the midpoint
                output_image[mid_x, mid_y] = [255, 0, 0]  # Set to blue
    
    # Save the output image
    cv2.imwrite(f"{z}_perimeter_1st_ex_ad.png", output_image)
    
    
    
    
    
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter_2nd_ex.png")
    
    # Threshold the image to isolate white grain boundary and blue pixels
    # Blue pixel range (B, G, R): (255, 0, 0)
    blue_pixels = np.where((image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))
    
    # Create a copy of the image to modify
    output_image = image.copy()
    
    # Define function to calculate Euclidean distance between two points
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    # Process each blue pixel
    for i in range(len(blue_pixels[0])):
        x1, y1 = blue_pixels[0][i], blue_pixels[1][i]
        
        for j in range(i + 1, len(blue_pixels[0])):
            x2, y2 = blue_pixels[0][j], blue_pixels[1][j]
            
            # Check if the two blue pixels are between 2-5 pixels apart
            distance = euclidean_distance((x1, y1), (x2, y2))
            if 1 <= distance <= 5:
                # Calculate the midpoint
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Remove the original blue pixels
                output_image[x1, y1] = [255, 255, 255]  # Set to white (original boundary color)
                output_image[x2, y2] = [255, 255, 255]  # Set to white (original boundary color)
                
                # Place the resultant blue pixel at the midpoint
                output_image[mid_x, mid_y] = [255, 0, 0]  # Set to blue
    
    # Save the output image
    cv2.imwrite(f"{z}_perimeter_2nd_ex_ad.png", output_image)
    
    
    
    
    
    #############Extracting only blue 2nd condition
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter_2nd_ex_ad.png")
    
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for the blue color in HSV
    lower_blue = np.array([100, 150, 0])  # Lower bound for blue
    upper_blue = np.array([140, 255, 255])  # Upper bound for blue
    
    # Create a mask to extract blue regions
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Create a black background
    output_image = np.zeros_like(image)
    
    # Extract only the blue parts of the image and set the rest to black
    output_image[mask > 0] = image[mask > 0]
    
    # Save the result
    cv2.imwrite(f"{z}_perimeter_2nd_ex_ad_ex.png", output_image)
    
    
    
    
    
    ############ combined extractions
    
    import cv2
    import numpy as np
    
    # Load the two images
    image1 = cv2.imread(f"{z}_perimeter_1st_ex_ad.png")  # First image
    image2 = cv2.imread(f"{z}_perimeter_2nd_ex_ad_ex.png")  # Second image
    
    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        print("The images must have the same dimensions.")
        exit()
    
    # Define the blue color in BGR format
    blue = [255, 0, 0]  # Blue color in OpenCV (BGR format)
    
    # Create a mask to find the blue pixels in the second image
    blue_mask = (image2 == blue).all(axis=-1)
    
    # Add the blue pixels from the second image to the first image
    image1[blue_mask] = blue
    
    # Save the result
    cv2.imwrite(f"{z}_perimeter_1st_2nd.png", image1)
    
    
    
    
    ################## adjusting blue pixels
    
    import cv2
    import numpy as np
    
    # Load the image
    image = cv2.imread(f"{z}_perimeter_1st_2nd.png")
    
    # Threshold the image to isolate white grain boundary and blue pixels
    # Blue pixel range (B, G, R): (255, 0, 0)
    blue_pixels = np.where((image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))
    
    # Create a copy of the image to modify
    output_image = image.copy()
    
    # Define function to calculate Euclidean distance between two points
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    # Process each blue pixel
    for i in range(len(blue_pixels[0])):
        x1, y1 = blue_pixels[0][i], blue_pixels[1][i]
        
        for j in range(i + 1, len(blue_pixels[0])):
            x2, y2 = blue_pixels[0][j], blue_pixels[1][j]
            
            # Check if the two blue pixels are between 2-5 pixels apart
            distance = euclidean_distance((x1, y1), (x2, y2))
            if 1 <= distance <= 5:
                # Calculate the midpoint
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Remove the original blue pixels
                output_image[x1, y1] = [255, 255, 255]  # Set to white (original boundary color)
                output_image[x2, y2] = [255, 255, 255]  # Set to white (original boundary color)
                
                # Place the resultant blue pixel at the midpoint
                output_image[mid_x, mid_y] = [255, 0, 0]  # Set to blue
    
    # Save the output image
    cv2.imwrite(f"{z}_perimeter_1st_2nd_ad.png", output_image)
    
    cv2.imshow("adjusted_image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    condition=int(input("Do you wish to edit the identified vertex points in the image then enter 0 otherwise enter 1 : "))
    if condition==0:
        
        
        import cv2
        import numpy as np

        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad.png")
        original_image = image.copy()  # Keep a copy of the original image for reset

        # Zoom parameters
        zoom_factor = 1.0
        zoom_step = 0.1
        min_zoom = 0.5
        max_zoom = 3.0

        # Flags and settings
        drawing = False  # True if the mouse is pressed
        erasing = False  # Toggle between drawing and erasing
        brush_size = 2  # Brush size for drawing or erasing
        color = (255, 255, 255)  # Default color (white)
        undo_stack = [image.copy()]  # Stack to store undo history

        # Mouse callback function
        def draw(event, x, y, flags, param):
            global drawing, erasing, image, zoom_factor
            
            # Convert zoomed coordinates back to the original coordinates
            zoom_x = int(x / zoom_factor)
            zoom_y = int(y / zoom_factor)
            
            # Start drawing/erasing when left mouse button is pressed
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                undo_stack.append(image.copy())  # Save the current image state for undo
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
            
            # Draw or erase as the mouse moves
            if event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    if not erasing:  # Drawing mode
                        cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                    else:  # Erasing mode
                        cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

        # Function to apply zoom
        def apply_zoom(image, zoom_factor):
            h, w = image.shape[:2]
            zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
            return zoomed_image

        # Trackbar callback function for brush size
        def set_brush_size(val):
            global brush_size
            brush_size = val

        # Trackbar callback function for color selection
        def set_color(val):
            global color
            if val == 0:
                color = (255, 0, 0)  # Blue
            elif val == 1:
                color = (255, 255, 255)  # White

        # Create a window and set mouse callback
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', draw)

        # Create trackbars for brush size and color selection
        cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
        cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

        while True:
            # Apply zoom to the image before displaying
            zoomed_image = apply_zoom(image, zoom_factor)
            cv2.imshow('Image', zoomed_image)
            
            # Wait for keypress
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('e'):  # Press 'e' to toggle erasing mode
                erasing = not erasing
                print('Erasing mode:', erasing)
            
            elif key == ord('r'):  # Press 'r' to reset the image
                image = original_image.copy()
                undo_stack.clear()
                undo_stack.append(image.copy())
                print('Image reset')
            
            elif key == ord('q'):  # Press 'q' to quit and save the image
                # Save the image before quitting
                cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit.png", image)
                break
            
            elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
                zoom_factor = min(zoom_factor + zoom_step, max_zoom)
                print('Zooming in:', zoom_factor)
            
            elif key == ord('-'):  # Press '-' to zoom out
                zoom_factor = max(zoom_factor - zoom_step, min_zoom)
                print('Zooming out:', zoom_factor)
            
            elif key == ord('u'):  # Press 'u' for undo
                if len(undo_stack) > 1:
                    undo_stack.pop()  # Remove the current state
                    image = undo_stack[-1].copy()  # Restore the previous state
                    print('Undo last action')

        # Close the window
        cv2.destroyAllWindows()
        
        
        #######important RED FILLING
        import cv2
        import numpy as np

        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit.png") 

        # Define color ranges (BGR format for OpenCV)
        blue_lower = np.array([100, 0, 0])    # Lower bound for blue
        blue_upper = np.array([255, 50, 50])  # Upper bound for blue
        white_lower = np.array([200, 200, 200])  # Lower bound for white
        white_upper = np.array([255, 255, 255])  # Upper bound for white

        # Find blue pixels
        blue_mask = cv2.inRange(image, blue_lower, blue_upper)

        # Find white pixels
        white_mask = cv2.inRange(image, white_lower, white_upper)

        # Temporary mask to mark red areas
        red_mask = np.zeros_like(blue_mask)

        # Get coordinates of blue pixels
        blue_pixels = np.argwhere(blue_mask > 0)

        # Set a radius of 10 pixels around each blue pixel and find white pixels nearby
        for blue_pixel in blue_pixels:
            y, x = blue_pixel
            # Define the region of interest (ROI) around each blue pixel
            y_min = max(0, y - 6) 
            y_max = min(image.shape[0], y + 6) 
            x_min = max(0, x - 6) 
            x_max = min(image.shape[1], x + 6) 
            
            # Extract the ROI from the white mask
            roi = white_mask[y_min:y_max, x_min:x_max]
            
            # Find coordinates of white pixels in the ROI
            white_pixels_in_roi = np.argwhere(roi > 0)
            
            # Mark white pixels in red_mask with a value of 255 (to signify red)
            for wp in white_pixels_in_roi:
                roi_y, roi_x = wp
                red_mask[y_min + roi_y, x_min + roi_x] += 1

        # Now, fill red in the original image only in locations where red_mask == 1
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if red_mask[y, x] == 1:  # If no overlap, mark as red
                    image[y, x] = [0, 0, 255]  # Red in BGR

        # Save or display the modified image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Red in BGR
        
        # Save the modified image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill.png", image)
        
        condition=int(input("If you wish to make changes Enter ‘0’ otherwise Enter ‘1’: "))
        if condition==0:
            import cv2
            import numpy as np

            image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit_fill.png")
            original_image = image.copy()  # Keep a copy of the original image for reset

            # Zoom parameters
            zoom_factor = 1.0
            zoom_step = 0.1
            min_zoom = 0.5
            max_zoom = 3.0

            # Flags and settings
            drawing = False  # True if the mouse is pressed
            erasing = False  # Toggle between drawing and erasing
            brush_size = 2  # Brush size for drawing or erasing
            color = (255, 255, 255)  # Default color (white)
            undo_stack = [image.copy()]  # Stack to store undo history

            # Mouse callback function
            def draw(event, x, y, flags, param):
                global drawing, erasing, image, zoom_factor
                
                # Convert zoomed coordinates back to the original coordinates
                zoom_x = int(x / zoom_factor)
                zoom_y = int(y / zoom_factor)
                
                # Start drawing/erasing when left mouse button is pressed
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    undo_stack.append(image.copy())  # Save the current image state for undo
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                
                # Draw or erase as the mouse moves
                if event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        if not erasing:  # Drawing mode
                            cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                        else:  # Erasing mode
                            cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

            # Function to apply zoom
            def apply_zoom(image, zoom_factor):
                h, w = image.shape[:2]
                zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
                return zoomed_image

            # Trackbar callback function for brush size
            def set_brush_size(val):
                global brush_size
                brush_size = val

            # Trackbar callback function for color selection
            def set_color(val):
                global color
                if val == 0:
                    color = (0, 0, 0)  # Black
                elif val == 1:
                    color = (255, 255, 255)  # White

            # Create a window and set mouse callback
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', draw)

            # Create trackbars for brush size and color selection
            cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
            cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

            while True:
                # Apply zoom to the image before displaying
                zoomed_image = apply_zoom(image, zoom_factor)
                cv2.imshow('Image', zoomed_image)
                
                # Wait for keypress
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('e'):  # Press 'e' to toggle erasing mode
                    erasing = not erasing
                    print('Erasing mode:', erasing)
                
                elif key == ord('r'):  # Press 'r' to reset the image
                    image = original_image.copy()
                    undo_stack.clear()
                    undo_stack.append(image.copy())
                    print('Image reset')
                
                elif key == ord('q'):  # Press 'q' to quit and save the image
                    # Save the image before quitting
                    cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill.png", image)
                    break
                
                elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
                    zoom_factor = min(zoom_factor + zoom_step, max_zoom)
                    print('Zooming in:', zoom_factor)
                
                elif key == ord('-'):  # Press '-' to zoom out
                    zoom_factor = max(zoom_factor - zoom_step, min_zoom)
                    print('Zooming out:', zoom_factor)
                
                elif key == ord('u'):  # Press 'u' for undo
                    if len(undo_stack) > 1:
                        undo_stack.pop()  # Remove the current state
                        image = undo_stack[-1].copy()  # Restore the previous state
                        print('Undo last action')

            # Close the window
            cv2.destroyAllWindows()
        
        else:
            print("vertex neighbours filled with red color")
        
        
        
        
        
        
        
        #######################imp extract blue and red (vertex blue and line red which form v shape)
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit_fill.png")
        
        # Convert the image to HSV (Hue, Saturation, Value) for better color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for the red color
        # Red can be tricky as it wraps around in the HSV color space, so we need two ranges
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        
        # Combine both red masks
        red_mask = mask1 + mask2
        
        # Define the lower and upper bounds for the blue color in HSV
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create a mask for the blue range
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        # Combine the red and blue masks
        combined_mask = red_mask + blue_mask
        
        # Create the output image, initializing with black background
        output_image = np.zeros_like(image)
        
        # Apply the combined mask to the original image, keeping only the red and blue pixels
        output_image[combined_mask > 0] = image[combined_mask > 0]
        
        # Save the result
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex.png", output_image)
        
        
        
        
        
        
        
        #############################DETECTING TWO SIDES OF V SHAPE
        
        
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex.png")
        
        # Define color ranges for red, blue, and black in BGR format
        blue_color = np.array([255, 0, 0])  # Blue color in BGR
        red_color = np.array([0, 0, 255])   # Red color in BGR
        
        # Create masks for blue and red
        blue_mask = np.all(image == blue_color, axis=-1)
        red_mask = np.all(image == red_color, axis=-1)
        
        # Directions for 8-connectivity neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        def trace_red_pixels(start_y, start_x, visited):
            queue = [(start_y, start_x)]
            end_pixels = set()
            while queue:
                current_y, current_x = queue.pop(0)
                if (current_y, current_x) not in visited:
                    visited.add((current_y, current_x))
                    is_end_pixel = True
                    # Explore neighbors to find red pixels
                    for dy, dx in directions:
                        ny, nx = current_y + dy, current_x + dx
                        if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:  # Check boundaries
                            if red_mask[ny, nx] and (ny, nx) not in visited:
                                queue.append((ny, nx))
                                is_end_pixel = False
                    if is_end_pixel:
                        end_pixels.add((current_y, current_x))
            return end_pixels
        
        # Iterate through blue pixels to find and trace connected red pixels
        visited = set()
        for blue_y, blue_x in zip(*np.where(blue_mask)):
            # Find all red pixels connected to this blue pixel
            for dy, dx in directions:
                red_y, red_x = blue_y + dy, blue_x + dx
                if 0 <= red_y < image.shape[0] and 0 <= red_x < image.shape[1]:
                    if red_mask[red_y, red_x] and (red_y, red_x) not in visited:
                        # Trace this red line and find all end pixels
                        end_pixels = trace_red_pixels(red_y, red_x, visited)
                        # Mark all end pixels as yellow
                        for end_y, end_x in end_pixels:
                            image[end_y, end_x] = [0, 255, 255]  # Yellow color in BGR
        
        # Save the modified image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex_all.png", image)
        
        
        
        
        
        
        
        
        #################Finding the angel by three points
        ######Angle restriction and average angles
        import cv2
        import numpy as np
        import math
        import pandas as pd

        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex_all.png")

        # Convert image to HSV color space
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for blue and yellow in HSV
        blue_lower = np.array([100, 100, 100])  # Adjust values as per the image
        blue_upper = np.array([140, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        # Create masks for blue and yellow pixels
        blue_mask = cv2.inRange(hsv_img, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

        # Find coordinates for blue and yellow pixels
        blue_coords = np.argwhere(blue_mask > 0)
        yellow_coords = np.argwhere(yellow_mask > 0)

        # Function to calculate the angle between two vectors
        def calculate_angle(p1, p2, vertex):
            v1 = p1 - vertex  # Vector from blue to first yellow
            v2 = p2 - vertex  # Vector from blue to second yellow
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi  # Convert to degrees
            return angle

        # Initialize vertex counter
        vertex_counter = 1

        # Initialize list for angles
        angles = []

        # Define angle restriction range (e.g., between 30 and 150 degrees)
        min_angle = int(input("min angle: "))
        max_angle = int(input("max angle: "))

        # Data storage for Excel file
        angle_details = []

        # Iterate over all blue pixels (vertices)
        for vertex in blue_coords:
            vertex = np.array(vertex[::-1])  # Convert to (x, y) format
            
            # Find the two closest yellow pixels to the blue vertex
            distances_to_yellow = [np.linalg.norm(np.array(y[::-1]) - vertex) for y in yellow_coords]
            if len(distances_to_yellow) < 2:
                continue  # Skip if less than two yellow pixels are found
            
            # Get the indices of the two closest yellow pixels
            closest_yellow_indices = np.argsort(distances_to_yellow)[:2]
            p1 = np.array(yellow_coords[closest_yellow_indices[0]][::-1])
            p2 = np.array(yellow_coords[closest_yellow_indices[1]][::-1])
            
            # Calculate the angle at the blue vertex
            angle = calculate_angle(p1, p2, vertex)
            
            # Check if the angle is within the defined range
            if min_angle <= angle <= max_angle:
                # Draw numbering on the image at the vertex
                cv2.putText(image, str(vertex_counter), tuple(vertex), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Print the angle
                print(f"Vertex {vertex_counter}: Angle = {angle:.2f} degrees")
                
                # Add the angle to the list
                angles.append(angle)
                
                # Store details in the list for Excel
                angle_details.append({"Vertex": vertex_counter, "Coordinates": tuple(vertex), "Angle (degrees)": angle})
                
                vertex_counter += 1

        # Calculate the average angle of all valid angles
        if angles:
            average_angle = np.mean(angles)
            std_dev33=np.std(angles)
        else:
            print("No angles found within the specified range.")

        # Save the results to an Excel file
        angle_df = pd.DataFrame(angle_details)
        angle_df.to_excel(f"{z}_angle_details.xlsx", index=False)

        # Save and display the image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_edit_fill_ex_all_angle.png", image)

        




        import pandas as pd
        import numpy as np
        from openpyxl import load_workbook

        # Load the Excel file
        file_name = f"{z}_angle_details.xlsx"
        sheet_name = "Sheet1"  # Adjust if the sheet has a different name
        data = pd.read_excel(file_name, sheet_name=sheet_name)

        # Assuming the angles are in a column named 'Angles'
        angle_data = data['Angle (degrees)']

        # Calculate the average and standard deviation
        average_angle33 = np.mean(angle_data)
        std_dev44= np.std(angle_data)

        # Load the workbook and select the sheet
        wb = load_workbook(file_name)
        sheet = wb[sheet_name]

        # Write the results to the Excel file (e.g., in cells D1 and D2)
        sheet["F1"] = "Average Angle "
        sheet["G1"] = f"{average_angle33:.2f} degrees "
        sheet["F2"] = "Standard Deviation"
        sheet["G2"] = f"{std_dev44:.2f} degrees"

        # Save the updated workbook
        wb.save(file_name)

        
        
        
        print("---------------------------------------------------------------------------------")
        
        
        print("The percentage of tungsten is :",(np.sum(seg2==1)*100)/(np.sum(seg2==1)+np.sum(seg2==0)))
        print("The percentage of Tungsten precipitates is :",(np.sum(seg222==1)*100)/(np.sum(seg222==1)+np.sum(seg222==0)))#true and false 0 ,1
        
        print(f"\nAverage grain size : {average_major_diameter:.2f} µm")
        print(f"Minimum Major Grain Diameter: {min_diameter:.2f} µm")
        print(f"Maximum Major Grain Diameter: {max_diameter:.2f} µm")
        print(f"Standard Deviation Grain Diameter: {std_dev:.2f} µm")


        print(f"Average Contiguity : {average_contiguity}")
        print(f"Standard Deviation of Contiguity: {std_dev1:.2f}")


        print(f"Average Angle: {average_angle:.2f} degrees")
        print(f"Standard Deviation of Angles: {std_dev33:.2f} degrees")
    
    
    
    else:
        #######important RED FILLING
        import cv2
        import numpy as np

        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad.png") 

        # Define color ranges (BGR format for OpenCV)
        blue_lower = np.array([100, 0, 0])    # Lower bound for blue
        blue_upper = np.array([255, 50, 50])  # Upper bound for blue
        white_lower = np.array([200, 200, 200])  # Lower bound for white
        white_upper = np.array([255, 255, 255])  # Upper bound for white

        # Find blue pixels
        blue_mask = cv2.inRange(image, blue_lower, blue_upper)

        # Find white pixels
        white_mask = cv2.inRange(image, white_lower, white_upper)

        # Temporary mask to mark red areas
        red_mask = np.zeros_like(blue_mask)

        # Get coordinates of blue pixels
        blue_pixels = np.argwhere(blue_mask > 0)

        # Set a radius of 10 pixels around each blue pixel and find white pixels nearby
        for blue_pixel in blue_pixels:
            y, x = blue_pixel
            # Define the region of interest (ROI) around each blue pixel
            y_min = max(0, y - 6) 
            y_max = min(image.shape[0], y + 6) 
            x_min = max(0, x - 6) 
            x_max = min(image.shape[1], x + 6) 
            
            # Extract the ROI from the white mask
            roi = white_mask[y_min:y_max, x_min:x_max]
            
            # Find coordinates of white pixels in the ROI
            white_pixels_in_roi = np.argwhere(roi > 0)
            
            # Mark white pixels in red_mask with a value of 255 (to signify red)
            for wp in white_pixels_in_roi:
                roi_y, roi_x = wp
                red_mask[y_min + roi_y, x_min + roi_x] += 1

        # Now, fill red in the original image only in locations where red_mask == 1
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if red_mask[y, x] == 1:  # If no overlap, mark as red
                    image[y, x] = [0, 0, 255]  # Red in BGR

        # Save or display the modified image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Red in BGR
        
        # Save the modified image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill.png", image)
        
        
        
        condition=int(input("If you wish to make changes Enter ‘0’ otherwise Enter ‘1’: "))
        if condition==0:
            import cv2
            import numpy as np

            image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_fill.png")
            original_image = image.copy()  # Keep a copy of the original image for reset

            # Zoom parameters
            zoom_factor = 1.0
            zoom_step = 0.1
            min_zoom = 0.5
            max_zoom = 3.0

            # Flags and settings
            drawing = False  # True if the mouse is pressed
            erasing = False  # Toggle between drawing and erasing
            brush_size = 2  # Brush size for drawing or erasing
            color = (255, 255, 255)  # Default color (white)
            undo_stack = [image.copy()]  # Stack to store undo history

            # Mouse callback function
            def draw(event, x, y, flags, param):
                global drawing, erasing, image, zoom_factor
                
                # Convert zoomed coordinates back to the original coordinates
                zoom_x = int(x / zoom_factor)
                zoom_y = int(y / zoom_factor)
                
                # Start drawing/erasing when left mouse button is pressed
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    undo_stack.append(image.copy())  # Save the current image state for undo
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                
                # Draw or erase as the mouse moves
                if event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        if not erasing:  # Drawing mode
                            cv2.circle(image, (zoom_x, zoom_y), brush_size, color, -1)  # Draw in selected color
                        else:  # Erasing mode
                            cv2.circle(image, (zoom_x, zoom_y), brush_size, original_image[zoom_y, zoom_x].tolist(), -1)  # Erase by restoring original pixel color

            # Function to apply zoom
            def apply_zoom(image, zoom_factor):
                h, w = image.shape[:2]
                zoomed_image = cv2.resize(image, (int(w * zoom_factor), int(h * zoom_factor)))
                return zoomed_image

            # Trackbar callback function for brush size
            def set_brush_size(val):
                global brush_size
                brush_size = val

            # Trackbar callback function for color selection
            def set_color(val):
                global color
                if val == 0:
                    color = (0, 0, 0)  # Black
                elif val == 1:
                    color = (255, 255, 255)  # White

            # Create a window and set mouse callback
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', draw)

            # Create trackbars for brush size and color selection
            cv2.createTrackbar('Brush Size', 'Image', brush_size, 20, set_brush_size)  # Brush size up to 20 pixels
            cv2.createTrackbar('Color', 'Image', 1, 1, set_color)  # Color: 0 for black, 1 for white

            while True:
                # Apply zoom to the image before displaying
                zoomed_image = apply_zoom(image, zoom_factor)
                cv2.imshow('Image', zoomed_image)
                
                # Wait for keypress
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('e'):  # Press 'e' to toggle erasing mode
                    erasing = not erasing
                    print('Erasing mode:', erasing)
                
                elif key == ord('r'):  # Press 'r' to reset the image
                    image = original_image.copy()
                    undo_stack.clear()
                    undo_stack.append(image.copy())
                    print('Image reset')
                
                elif key == ord('q'):  # Press 'q' to quit and save the image
                    # Save the image before quitting
                    cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill.png", image)
                    break
                
                elif key == ord('+') or key == ord('='):  # Press '+' to zoom in
                    zoom_factor = min(zoom_factor + zoom_step, max_zoom)
                    print('Zooming in:', zoom_factor)
                
                elif key == ord('-'):  # Press '-' to zoom out
                    zoom_factor = max(zoom_factor - zoom_step, min_zoom)
                    print('Zooming out:', zoom_factor)
                
                elif key == ord('u'):  # Press 'u' for undo
                    if len(undo_stack) > 1:
                        undo_stack.pop()  # Remove the current state
                        image = undo_stack[-1].copy()  # Restore the previous state
                        print('Undo last action')

            # Close the window
            cv2.destroyAllWindows()
        
        else:
            print("vertex neighbours filled with red color")
        
        
        
        
        
        #######################imp extract blue and red (vertex blue and line red which form v shape)
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_fill.png")
        
        # Convert the image to HSV (Hue, Saturation, Value) for better color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for the red color
        # Red can be tricky as it wraps around in the HSV color space, so we need two ranges
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        
        # Combine both red masks
        red_mask = mask1 + mask2
        
        # Define the lower and upper bounds for the blue color in HSV
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create a mask for the blue range
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        # Combine the red and blue masks
        combined_mask = red_mask + blue_mask
        
        # Create the output image, initializing with black background
        output_image = np.zeros_like(image)
        
        # Apply the combined mask to the original image, keeping only the red and blue pixels
        output_image[combined_mask > 0] = image[combined_mask > 0]
        
        # Save the result
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill_ex.png", output_image)
        
        
        
        
        
        
        
        
        #############################DETECTING TWO SIDES OF V SHAPE
        
        
        import cv2
        import numpy as np
        
        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_fill_ex.png")
        
        # Define color ranges for red, blue, and black in BGR format
        blue_color = np.array([255, 0, 0])  # Blue color in BGR
        red_color = np.array([0, 0, 255])   # Red color in BGR
        
        # Create masks for blue and red
        blue_mask = np.all(image == blue_color, axis=-1)
        red_mask = np.all(image == red_color, axis=-1)
        
        # Directions for 8-connectivity neighbors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        def trace_red_pixels(start_y, start_x, visited):
            queue = [(start_y, start_x)]
            end_pixels = set()
            while queue:
                current_y, current_x = queue.pop(0)
                if (current_y, current_x) not in visited:
                    visited.add((current_y, current_x))
                    is_end_pixel = True
                    # Explore neighbors to find red pixels
                    for dy, dx in directions:
                        ny, nx = current_y + dy, current_x + dx
                        if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:  # Check boundaries
                            if red_mask[ny, nx] and (ny, nx) not in visited:
                                queue.append((ny, nx))
                                is_end_pixel = False
                    if is_end_pixel:
                        end_pixels.add((current_y, current_x))
            return end_pixels
        
        # Iterate through blue pixels to find and trace connected red pixels
        visited = set()
        for blue_y, blue_x in zip(*np.where(blue_mask)):
            # Find all red pixels connected to this blue pixel
            for dy, dx in directions:
                red_y, red_x = blue_y + dy, blue_x + dx
                if 0 <= red_y < image.shape[0] and 0 <= red_x < image.shape[1]:
                    if red_mask[red_y, red_x] and (red_y, red_x) not in visited:
                        # Trace this red line and find all end pixels
                        end_pixels = trace_red_pixels(red_y, red_x, visited)
                        # Mark all end pixels as yellow
                        for end_y, end_x in end_pixels:
                            image[end_y, end_x] = [0, 255, 255]  # Yellow color in BGR
        
        # Save the modified image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill_ex_all.png", image)
        
        
        
        
        
        
        
        
        #################Finding the angel by three points
        ######Angle restriction and average angles
        import cv2
        import numpy as np
        import math
        import pandas as pd

        # Load the image
        image = cv2.imread(f"{z}_perimeter_1st_2nd_ad_fill_ex_all.png")

        # Convert image to HSV color space
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for blue and yellow in HSV
        blue_lower = np.array([100, 100, 100])  # Adjust values as per the image
        blue_upper = np.array([140, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        # Create masks for blue and yellow pixels
        blue_mask = cv2.inRange(hsv_img, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

        # Find coordinates for blue and yellow pixels
        blue_coords = np.argwhere(blue_mask > 0)
        yellow_coords = np.argwhere(yellow_mask > 0)

        # Function to calculate the angle between two vectors
        def calculate_angle(p1, p2, vertex):
            v1 = p1 - vertex  # Vector from blue to first yellow
            v2 = p2 - vertex  # Vector from blue to second yellow
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi  # Convert to degrees
            return angle

        # Initialize vertex counter
        vertex_counter = 1

        # Initialize list for angles
        angles = []

        # Define angle restriction range (e.g., between 30 and 150 degrees)
        min_angle = int(input("min angle: "))
        max_angle = int(input("max angle: "))

        # Data storage for Excel file
        angle_details = []

        # Iterate over all blue pixels (vertices)
        for vertex in blue_coords:
            vertex = np.array(vertex[::-1])  # Convert to (x, y) format
            
            # Find the two closest yellow pixels to the blue vertex
            distances_to_yellow = [np.linalg.norm(np.array(y[::-1]) - vertex) for y in yellow_coords]
            if len(distances_to_yellow) < 2:
                continue  # Skip if less than two yellow pixels are found
            
            # Get the indices of the two closest yellow pixels
            closest_yellow_indices = np.argsort(distances_to_yellow)[:2]
            p1 = np.array(yellow_coords[closest_yellow_indices[0]][::-1])
            p2 = np.array(yellow_coords[closest_yellow_indices[1]][::-1])
            
            # Calculate the angle at the blue vertex
            angle = calculate_angle(p1, p2, vertex)
            
            # Check if the angle is within the defined range
            if min_angle <= angle <= max_angle:
                # Draw numbering on the image at the vertex
                cv2.putText(image, str(vertex_counter), tuple(vertex), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Print the angle
                print(f"Vertex {vertex_counter}: Angle = {angle:.2f} degrees")
                
                # Add the angle to the list
                angles.append(angle)
                
                # Store details in the list for Excel
                angle_details.append({"Vertex": vertex_counter, "Coordinates": tuple(vertex), "Angle (degrees)": angle})
                
                vertex_counter += 1

        # Calculate the average angle of all valid angles
        if angles:
            average_angle = np.mean(angles)
            std_dev33=np.std(angles)
            print(f"Standard Deviation of Angles: {std_dev33:.2f} degrees")
            print(f"Average Angle: {average_angle:.2f} degrees")
        else:
            print("No angles found within the specified range.")

        # Save the results to an Excel file
        angle_df = pd.DataFrame(angle_details)
        angle_df.to_excel(f"{z}_angle_details.xlsx", index=False)

        # Save and display the image
        cv2.imwrite(f"{z}_perimeter_1st_2nd_ad_fill_ex_all_angle.png", image)

        




        import pandas as pd
        import numpy as np
        from openpyxl import load_workbook

        # Load the Excel file
        file_name = f"{z}_angle_details.xlsx"
        sheet_name = "Sheet1"  # Adjust if the sheet has a different name
        data = pd.read_excel(file_name, sheet_name=sheet_name)

        # Assuming the angles are in a column named 'Angles'
        angle_data = data['Angle (degrees)']

        # Calculate the average and standard deviation
        average_angle33 = np.mean(angle_data)
        std_dev44= np.std(angle_data)

        # Load the workbook and select the sheet
        wb = load_workbook(file_name)
        sheet = wb[sheet_name]

        # Write the results to the Excel file (e.g., in cells D1 and D2)
        sheet["F1"] = "Average Angle "
        sheet["G1"] = f"{average_angle33:.2f} degrees "
        sheet["F2"] = "Standard Deviation"
        sheet["G2"] = f"{std_dev44:.2f} degrees"

        # Save the updated workbook
        wb.save(file_name)

        
        
        
        
        print("---------------------------------------------------------------------------------")
        
        
        print("The percentage of tungsten is :",(np.sum(seg2==1)*100)/(np.sum(seg2==1)+np.sum(seg2==0)))
        print("The percentage of Tungsten precipitates is :",(np.sum(seg222==1)*100)/(np.sum(seg222==1)+np.sum(seg222==0)))#true and false 0 ,1
        
        
        print(f"\nAverage grain size: {average_major_diameter:.2f} µm")
        print(f"Minimum Major Grain Diameter: {min_diameter:.2f} µm")
        print(f"Maximum Major Grain Diameter: {max_diameter:.2f} µm")
        print(f"Standard Deviation Grain Diameter: {std_dev:.2f} µm")


        print(f"Average Contiguity : {average_contiguity}")
        print(f"Standard Deviation of Contiguity: {std_dev1:.2f}")


        print(f"Average Angle: {average_angle:.2f} degrees")
        print(f"Standard Deviation of Angles: {std_dev33:.2f} degrees")