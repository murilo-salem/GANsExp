import cv2 as cv
import numpy as np
import rasterio

with rasterio.open('/home/lucas-fontoura/Documents/Pix2Pix/Data/pix2pix_full_cornfield_renormalized_256/train/target/patch_42.tif', ) as src:
    img = src.read()
    profile = src.profile

gray_img = cv.cvtColor(img.transpose(1, 2 ,0), cv.COLOR_RGB2GRAY)

# Desired depth for the output image (e.g., cv.CV_64F to avoid data loss)
ddepth = cv.CV_64F 

# Calculate the gradient in the X direction
grad_x = cv.Sobel(src=gray_img, ddepth=ddepth, dx=1, dy=0, ksize=3) #

# Calculate the gradient in the Y direction
grad_y = cv.Sobel(src=gray_img, ddepth=ddepth, dx=0, dy=1, ksize=3)

grad_norm = np.sqrt(grad_x**2 + grad_y**2)


# breakpoint()

cv.imwrite("grad_norm.bmp", grad_norm)
# with rasterio.open("grad_norm.tif", "w", **profile) as dst:
#     dst.write(grad_norm)