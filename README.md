# Low-Light Photography Enhancement

This project focuses on enhancing low-light photographs by combining the strengths of flash and no-flash images using advanced image processing techniques.

## Steps for Enhancement

### 1. Initial Processing
- Convert the image from BGR to RGB.
- Change pixel values from integers to doubles and normalize by dividing by 255.

### 2. Shadow Mask Application
- Apply a shadow mask to preserve shadows.

#### Shadow Mask Algorithm:
1. Compute the luminescence (brightness) of the image.
2. Identify significantly brighter areas by calculating the difference.
3. Use a binary matrix (0-1 matrix) to mark potential shadow spots.
4. Use morphological functions (flood fill, erode, dilation) to clean the mask.
5. Smooth the mask edges with a Gaussian filter.
- Return the mask.

### 3. Color Component Separation
- Split flash and no-flash images into their respective R, G, and B components.

### 4. Bilateral Filter Computation
- Compute bilateral filters for both flash and no-flash images for each color component separately.

#### Bilateral Filter Algorithm:
1. Define parameters: standard deviations and kernel size.
   - `s_s` (spatial effect) assigns weights based on distances.
   - `s_r` (intensity effect) assigns weights based on intensity differences; smaller values preserve edges better.
   - `ws` (kernel size).
2. Create a Gaussian filter with size `ws` and standard deviation `s_s`, and pad images with half the filter size.
3. Define outputs: joint bilateral filter (JBF), flash base, and no-flash base.
4. Iterate over the image (for both flash and no-flash images):
   - Extract a square segment of the image matching the filter size.
   - Subtract the center pixel value to simplify the code.
   - Create an intensity mask for the segment to assign weights based on intensities (using `s_r`).
   - Combine the Gaussian filter with intensity weights to form masks for both flash and no-flash images (normalize them).
   - Apply the flash mask to the no-flash segment to create a joint no-flash mask.
   - Apply the flash mask to the flash segment (producing flash base) and the no-flash mask to the no-flash segment (producing no-flash base).
   - Sum the matrix and assign the value to the center pixel.
5. Return the joint no-flash, flash base, and no-flash base images.

### 5. Component Stacking
- Stack the individual color components to form the joint image, flash base, and ambient base images.
- Divide the flash image by the flash base image to obtain flash details (high-frequency parts).

#### Summary of Products:
1. Shadow mask
2. Flash details
3. No-flash base (ambient base)
4. Joint image

### 6. Final Image Composition
- Combine the processed components using the formula:
  final_image = (1 - shadow_mask) * joint_image * flash_details + shadow_mask * ambient_base

### 7. Additional Processing
- Further shadow filtering may be required as shadows caused by the flash might receive less weight.
- Flash details help recover details lost in the no-flash image.
