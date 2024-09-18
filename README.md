# Cyclopean-Image-Creation


1. Two Stereoscopic Images (Left and Right View):

    The process starts with two stereoscopic images — one from the left eye view and the other from the right eye view.

2. Convert to HSV Format:

    Convert the stereo images from RGB to HSV format, as HSV can sometimes make image processing tasks (especially those related to color) more intuitive by separating chromatic and intensity components.

3. Compute Gradients (in Both X and Y Directions) for Left and Right Images:

    Gradients in the x (horizontal) and y (vertical) directions are computed for both the left and right images in the HSV color channels (hue, saturation, value).

4. Compute the Z Component:

    For each pixel, subtract the pixel value in the right image from the corresponding pixel in the left image:
    Iiz=ILi−IRi
    Iiz​=ILi​​−IRi​​ where ILiILi​​ and IRiIRi​​ are the HSV values for the left and right images respectively. This z-component represents the disparity between the two views for that particular color channel (hue, saturation, or value).

5. Create Extended Tensor:

    Create the extended structure tensor at each pixel. The tensor includes terms like:
    S=[xxxyxzyxyyyzzxzyzz]
    S=

​xxyxzx​xyyyzy​xzyzzz​
Where xx, yy, xy, etc., are the products of the gradients in the x, y, and z directions and their interactions. These terms combine the local intensity variations (gradients) within the left image and the disparity between the left and right images (via the z-component).

6. 3x3 Tensor at Each Pixel:

    At each pixel, you compute a 3x3 matrix (extended tensor) that encapsulates the local geometry (changes in intensity in the x and y directions) and the disparity between the left and right images.

7. Compute Eigenvalues and Eigenvectors:

    For each pixel, compute the eigenvalues and eigenvectors of the 3x3 extended tensor. The largest eigenvalue and its corresponding eigenvector give you the dominant direction and magnitude of change at that pixel.

8. Take the Eigenvalue Corresponding to the Largest Eigenvector:

    Identify the largest eigenvalue because it represents the strongest change or disparity at that pixel. The corresponding eigenvector provides the direction of that change.

9. Disparity Map Computation:

    By repeating this process for each pixel, you construct a disparity map that indicates how much the left and right images differ (the disparity) at every pixel.

10. Subtract Disparity Map from Right Image:

    The operation described is not typically element-wise subtraction between the disparity map and the right image. Instead, the disparity map tells you how much to shift each pixel in the right image to align it with the left image.
    The idea is to use the disparity map to adjust or warp the right image so that corresponding points in the left and right views are aligned.

11. Cyclopean Image Creation:

    Once the right image is aligned with the left image using the disparity map, extract corresponding patches from both the left image and the aligned right image. You then average these patches to create the cyclopean image, which simulates what the human brain perceives when viewing the stereo pair.
