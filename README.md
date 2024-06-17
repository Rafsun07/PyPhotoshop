# PyPhotoshop
PyPhotoshop is a Python library designed for performing various image processing tasks, including brightening, contrast adjustment, blurring, edge detection, and image combination.

## Features

- Brighten/Darken Images: Adjust the brightness of images.
- Contrast Adjustment: Modify the contrast of images.
- Blurring: Apply a blur effect to images with customizable kernel size.
- Edge Detection: Detect edges using kernels like Sobel filters.
- Image Combination: Combine two images using a squared sum of squares method.

## Installation

1. **Clone the repository:**

    ```
    git clone https://github.com/Rafsun07/PyPhotoshop.git
    cd PyPhotoshop
    ```

2. **Run the solver:**

    ```
    python PyPhotoshop.py
    ```

## Usage

**To use the PyPhotoshop, follow these instructions:**

   ```
    from image import Image
    import numpy as np
    from processing import brighten, adjust_contrast, blur, apply_kernel, combine_images

    # Load an image
    image = Image(filename='lake.png')

    # Brighten the image
    brightened_image = brighten(image, 1.7)
    brightened_image.write_image('brightened.png')

    # Increase contrast
    contrast_image = adjust_contrast(image, 2, 0.5)
    contrast_image.write_image('contrast.png')

    # Apply a blur with a kernel size of 3
    blurred_image = blur(image, 3)
    blurred_image.write_image('blurred.png')

    # Apply a Sobel edge detection kernel
    sobel_x_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    edge_image = apply_kernel(image, sobel_x_kernel)
    edge_image.write_image('edge.png')

    # Combine two images
    image1 = Image(filename='lake.png')
    image2 = Image(filename='city.png')
    combined_image = combine_images(image1, image2)
    combined_image.write_image('combined.png')
    
   ```

## How the Code Works

### Image Class:
   
 The `Image` class is designed to handle image loading, saving, and basic manipulations. Here's a breakdown of its components:

 **Initialization (`__init__` method):**
 
 The constructor initializes an image object. You can either create an image from a file or create a blank image with specified dimensions.
   

  ```
    def __init__(self, x_pixels=0, y_pixels=0, num_channels=0, filename=''):
        self.input_path = 'input/'
        self.output_path = 'output/'
        if x_pixels and y_pixels and num_channels:
            self.x_pixels = x_pixels
            self.y_pixels = y_pixels
            self.num_channels = num_channels
            self.array = np.zeros((x_pixels, y_pixels, num_channels))
        elif filename:
            self.array = self.read_image(filename)
            self.x_pixels, self.y_pixels, self.num_channels = self.array.shape
        else:
            raise ValueError("You need to input either a filename OR specify the dimensions of the image")
  ```

**Reading an Image (`read_image` method):**

This method reads an image from a file, applies gamma correction, and converts it to a numpy array.

   ```
    def read_image(self, filename, gamma=2.2):
        im = png.Reader(self.input_path + filename).asFloat()
        resized_image = np.vstack(list(im[2]))
        resized_image.resize(im[1], im[0], 3)
        resized_image = resized_image ** gamma
        return resized_image
   ```

**Writing an Image (`write_image` method):**

This method writes the image data to a file after applying gamma correction.

  ```
    def write_image(self, output_file_name, gamma=2.2):
        im = np.clip(self.array, 0, 1)
        y, x = self.array.shape[0], self.array.shape[1]
        im = im.reshape(y, x*3)
        writer = png.Writer(x, y)
        with open(self.output_path + output_file_name, 'wb') as f:
            writer.write(f, 255*(im**(1/gamma)))
        self.array.resize(y, x, 3)def write_image(self, output_file_name, gamma=2.2):
        im = np.clip(self.array, 0, 1)
        y, x = self.array.shape[0], self.array.shape[1]
        im = im.reshape(y, x*3)
        writer = png.Writer(x, y)
        with open(self.output_path + output_file_name, 'wb') as f:
            writer.write(f, 255*(im**(1/gamma)))
        self.array.resize(y, x, 3)
  ```
### Image Processing Functions:

Several functions perform different image processing tasks using the `Image` class.

**Brighten:**
 
This function increases the brightness of the image.
   

  ```
    def brighten(image, factor):
        new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
        new_im.array = image.array * factor
        return new_im
  ```

**Adjust Contrast:**
 
This function adjusts the contrast of the image by changing the difference from a midpoint.
   

  ```
    def adjust_contrast(image, factor, mid):
        new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
        for x in range(image.x_pixels):
            for y in range(image.y_pixels):
                for c in range(image.num_channels):
                    new_im.array[x, y, c] = (image.array[x, y, c] - mid) * factor + mid
        return new_im
  ```

**Blur:**
 
This function applies a blur effect using a simple average of neighboring pixels.
   

  ```
    def blur(image, kernel_size):
         new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
         neighbor_range = kernel_size // 2
         for x in range(image.x_pixels):
             for y in range(image.y_pixels):
                 for c in range(image.num_channels):
                    total = 0
                    for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                        for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                            total += image.array[x_i, y_i, c]
                    new_im.array[x, y, c] = total / (kernel_size ** 2)
        return new_im
  ```

**Apply Kernel:**

This function applies a convolutional kernel to the image, useful for edge detection.
   

  ```
    def apply_kernel(image, kernel):
        new_im = Image(x_pixels=image.x_pixels, y_pixels=image.y_pixels, num_channels=image.num_channels)
        neighbor_range = kernel.shape[0] // 2
        for x in range(image.x_pixels):
             for y in range(image.y_pixels):
                 for c in range(image.num_channels):
                     total = 0
                     for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                         for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                            x_k = x_i + neighbor_range - x
                            y_k = y_i + neighbor_range - y
                            kernel_val = kernel[x_k, y_k]
                            total += image.array[x_i, y_i, c] * kernel_val
                     new_im.array[x, y, c] = total
        return new_im
  ```

**Combine Images:**

This function combines two images using a squared sum of squares method.
   

  ```
    def combine_images(image1, image2):
        new_im = Image(x_pixels=image1.x_pixels, y_pixels=image1.y_pixels, num_channels=image1.num_channels)
        for x in range(image1.x_pixels):
             for y in range(image1.y_pixels):
                 for c in range(image1.num_channels):
                     new_im.array[x, y, c] = (image1.array[x, y, c]**2 + image2.array[x, y, c]**2)**0.5
        return new_im
  ```

### Main Script

The main script demonstrates how to use the Image class and processing functions.

```
    if __name__ == '__main__':
         lake = Image(filename='lake.png')
         city = Image(filename='city.png')

         brightened_im = brighten(lake, 1.7)
         brightened_im.write_image('brightened.png')

         darkened_im = brighten(lake, 0.3)
         darkened_im.write_image('darkened.png')

         incr_contrast = adjust_contrast(lake, 2, 0.5)
         incr_contrast.write_image('increased_contrast.png')

         decr_contrast = adjust_contrast(lake, 0.5, 0.5)
         decr_contrast.write_image('decreased_contrast.png')

         blur_3 = blur(city, 3)
         blur_3.write_image('blur_k3.png')

         blur_15 = blur(city, 15)
         blur_15.write_image('blur_k15.png')

         sobel_x = apply_kernel(city, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
         sobel_x.write_image('edge_x.png')
         sobel_y = apply_kernel(city, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
         sobel_y.write_image('edge_y.png')

         sobel_xy = combine_images(sobel_x, sobel_y)
         sobel_xy.write_image('edge_xy.png')

  ```

### Summary

- The `Image` class handles loading, saving, and manipulating image data.
- Various functions perform specific image processing tasks like brightening, contrast adjustment, blurring, applying convolutional kernels, and combining images.
- The main script demonstrates these functions using example images.

## FAQ

Q: What image formats are supported?

A: Currently, only PNG images are supported.

Q: Can I use this toolkit with grayscale images?

A: Yes, but some functions are tailored for RGB images.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the project maintainer at [rafsun.eram@gmail.com](mailto:rafsun.eram@gmail.com).

## Acknowledgements

- Developed using Python.
- Inspired by Photoshop.

---
