import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import albumentations as A

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.augmented_images = None

        # Define button_frame as an instance variable
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.LEFT, padx=10)

        # Create and set up widgets
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons and graph
        button_graph_frame = tk.Frame(self.root)
        button_graph_frame.pack(padx=10, pady=90)

        # Frame for buttons on the left
        button_width = 30
        button_height = 3
        button_font_size = 15  

        # Load Image button
        self.load_button = tk.Button(button_graph_frame, text="Load Image", command=self.load_image, height=button_height, width=button_width, font=("Arial", button_font_size))
        self.load_button.grid(row=0, column=0, pady=20, padx=10)

        # Combine Process button
        self.combine_process_button = tk.Button(button_graph_frame, text="Generate Final Image", command=self.combine_process, height=button_height, width=button_width, font=("Arial", button_font_size))
        self.combine_process_button.grid(row=1, column=0, pady=20, padx=10)

        # Matplotlib graph for the main window
        figure_width = 6
        figure_height = 6
        self.figure_main, self.ax_main = plt.subplots(figsize=(figure_width, figure_height))
        self.graph_canvas_main = FigureCanvasTkAgg(self.figure_main, master=button_graph_frame)
        self.graph_canvas_main.get_tk_widget().grid(row=0, column=1, rowspan=2, padx=20)  # Adjust padx as needed

        # Frame for additional buttons below the graph
        additional_buttons_frame = tk.Frame(self.root)
        additional_buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Label for optional processes
        optional_label = tk.Label(additional_buttons_frame, text="Optional Images:", font=("Arial", button_font_size, "underline"))
        optional_label.pack(side=tk.TOP, pady=15)

        # Additional Buttons
        additional_buttons = [
            ("Generate Gaussian Image", self.process_image),
            ("Generate Augmented Images", self.augment_image),
            ("Generate K-Space Truncation Images", self.k_space_truncation)
        ]

        # Create a new frame for additional buttons
        additional_buttons_row_frame = tk.Frame(additional_buttons_frame)
        additional_buttons_row_frame.pack(side=tk.TOP, fill=tk.X)

        # Adjust button height to make them smaller
        button_height_additional = 2

       # Add buttons with expand and fill options
        for text, command in additional_buttons:
         button = tk.Button(additional_buttons_row_frame, text=text, command=command, height=button_height_additional, font=("Arial", button_font_size))
         button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Explicitly update the Matplotlib plot
        self.graph_canvas_main.draw()


    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(self.image_path)
            img_array = np.array(self.original_image)

            # Display the loaded image in the main window graph
            self.ax_main.clear()
            self.ax_main.imshow(img_array, cmap='gray')
            self.graph_canvas_main.draw()

    def process_image(self):
        if self.image_path:
            img = cv2.imread(self.image_path, cv2.COLOR_BGR2RGB)
            img_shape = img.shape

            # Applying Gaussian noise
            mean_value = 120
            std_deviation_value = 30
            gauss_noise = self.gauss_noise_calculator(mean_value, std_deviation_value, *img_shape)

            # Adding noise to the original image
            self.processed_image = cv2.add(img, gauss_noise)

            # Display the processed image in a new window with its own graph
            self.display_in_new_window(np.array(self.processed_image))

    def augment_image(self):
     if self.image_path:
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transforms = [
            ("CLAHE", A.CLAHE()),
            ("VerticalFlip", A.VerticalFlip()),
            ("HorizontalFlip", A.HorizontalFlip()),
            ("RandomRotate90", A.RandomRotate90()),
            ("Transpose", A.Transpose()),
            ("GridDistortion", A.GridDistortion())
        ]

        augmented_images = []

        for transform_name, transform in transforms:
            augmented_image = transform(image=img)['image']
            augmented_images.append((transform_name, augmented_image))

        # Display the augmented images in the same window
        self.display_augmented_images(augmented_images)

    def display_augmented_images(self, augmented_images):
        new_window = tk.Toplevel(self.root)
        new_window.title("Augmented Images")

        img_width, img_height = self.original_image.size  # Same size as the original image
        img_spacing = 20  # Increased spacing between images

        # Calculate the canvas size based on the number of images and spacing
        canvas_width = 4 * img_width + 3 * img_spacing
        canvas_height = ((len(augmented_images) - 1) // 4 + 1) * (img_height + img_spacing)

        canvas = tk.Canvas(new_window, width=canvas_width, height=canvas_height)
        canvas.pack()

        image_refs = []  # To keep references to the images

        for i, (transform_name, augmented_image) in enumerate(augmented_images):
            augmented_image = Image.fromarray(augmented_image).resize((img_width, img_height))

            img = ImageTk.PhotoImage(augmented_image)
            x_position = (i % 4) * (img_width + img_spacing)
            y_position = (i // 4) * (img_height + img_spacing)

            canvas.create_image(x_position, y_position, anchor=tk.NW, image=img)
            canvas.create_text(x_position + img_width // 2, y_position + img_height + 10, text=transform_name)

            image_refs.append(img)

            # Store references to images to avoid garbage collection
            canvas.image_refs = image_refs

    def downsample_via_kspace_truncation(self, image, downsampling_factor, blur_sigma=1):
        # Convert to numpy array
        image_array = np.array(image)

        # Apply Gaussian blur for noise reduction
        blurred_image_array = cv2.GaussianBlur(image_array, (0, 0), blur_sigma)

        # Calculate the truncation radius
        k_space = fftshift(fft2(blurred_image_array))
        truncation_radius = self
    def downsample_via_kspace_truncation(self, image, downsampling_factor, blur_sigma=1):
        # Convert to numpy array
        image_array = np.array(image)

        # Apply Gaussian blur for noise reduction
        blurred_image_array = cv2.GaussianBlur(image_array, (0, 0), blur_sigma)

        # Calculate the truncation radius
        k_space = fftshift(fft2(blurred_image_array))
        truncation_radius = self.find_truncation_radius(k_space)

        # Perform k-space truncation
        k_space = fftshift(fft2(blurred_image_array))
        center_x, center_y = np.array(k_space.shape) // 2
        y, x = np.ogrid[:k_space.shape[0], :k_space.shape[1]]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= truncation_radius ** 2
        k_space[~mask] = 0
        truncated_image = ifft2(ifftshift(k_space)).real

        # Resize (downsample) the image
        downsampled_image = Image.fromarray(np.clip(truncated_image, 0, 255).astype(np.uint8))
        downsampled_image = downsampled_image.resize(
            (image_array.shape[1] // downsampling_factor, image_array.shape[0] // downsampling_factor),
            Image.LANCZOS
        )

        return downsampled_image

    def find_truncation_radius(self, k_space, energy_threshold=0.95):
        """Find a truncation radius based on the energy distribution in k-space."""
        magnitude_spectrum = np.abs(k_space)
        total_energy = np.sum(magnitude_spectrum)
        center_x, center_y = np.array(k_space.shape) // 2
        y, x = np.ogrid[:k_space.shape[0], :k_space.shape[1]]
        radii = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        sorted_indices = np.argsort(radii.ravel())
        cumulative_energy = np.cumsum(magnitude_spectrum.ravel()[sorted_indices])
        threshold_energy = total_energy * energy_threshold
        radius_index = np.searchsorted(cumulative_energy, threshold_energy)
        return radii.ravel()[sorted_indices[radius_index]]

    @staticmethod
    def gauss_noise_calculator(mean_value, std_deviation_value, img_x_size, img_y_size):
        gauss_noise = np.zeros((img_x_size, img_y_size), dtype=np.uint8)
        cv2.randn(gauss_noise, mean_value, std_deviation_value)
        gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
        return gauss_noise
    
    def k_space_truncation(self):
        if self.image_path:
            img = Image.open(self.image_path).convert('L')
            downsampled_images = []

            for factor in [2, 4, 8]:
                downsampled_image = self.downsample_via_kspace_truncation(img, factor)
                downsampled_images.append(downsampled_image)

            # Display all three k-space truncation images in a new window
            self.display_multiple_images(downsampled_images)

    def combine_process(self):
        if self.image_path:
            img = Image.open(self.image_path).convert('L')
            processed_images_with_noise = []

            for factor in [2, 4, 8]:
                # Apply Gaussian noise to the original image
                img_array = np.array(img)
                mean_value = 120
                std_deviation_value = 30
                gauss_noise = self.gauss_noise_calculator(mean_value, std_deviation_value, *img_array.shape)
                img_with_noise = cv2.add(img_array, gauss_noise)

                # Perform k-space truncation with added Gaussian noise
                processed_image_with_noise = self.downsample_via_kspace_truncation(img_with_noise, factor)
                processed_images_with_noise.append(processed_image_with_noise)

            # Display all three k-space truncation images with added Gaussian noise in a new window
            self.display_multiple_images(processed_images_with_noise, title="Combined Process")

    def display_in_new_window(self, image, title="Processed Image"):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)

        figure_new, ax_new = plt.subplots(figsize=(6, 4))
        graph_canvas_new = FigureCanvasTkAgg(figure_new, master=new_window)
        graph_canvas_new.get_tk_widget().pack()

        ax_new.imshow(image, cmap='gray')  # Use cmap='gray' for grayscale images
        graph_canvas_new.draw()
    
    def display_multiple_images(self, images, title=None):
        new_window = tk.Toplevel(self.root)
        if title:
            new_window.title(title)

        img_width, img_height = self.original_image.size
        img_spacing = 20

        # Calculate the canvas size based on the number of images and spacing
        canvas_width = len(images) * (img_width + img_spacing)
        canvas_height = img_height

        canvas = tk.Canvas(new_window, width=canvas_width, height=canvas_height)
        canvas.pack()

        image_refs = []

        for i, downsampled_image in enumerate(images):
            downsampled_image = Image.fromarray(np.array(downsampled_image))

            img = ImageTk.PhotoImage(downsampled_image)
            x_position = i * (img_width + img_spacing)
            y_position = 0

            canvas.create_image(x_position, y_position, anchor=tk.NW, image=img)
            canvas.create_text(x_position + img_width // 2, y_position + img_height + 10, text=f"Factor {2 ** (i + 1)}")

            image_refs.append(img)

        # Store references to images to avoid garbage collection
        canvas.image_refs = image_refs

    
def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)

    root.geometry("1920x1080")  

    root.mainloop()

if __name__ == "__main__":
    main()
