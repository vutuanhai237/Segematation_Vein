import cv2
import numpy as np
# 1. Gray scale
origin = cv2.imread('Image/13.jpg', 1)
img = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)
cv2.imshow("test1", origin)
# gray = (((h + 90) % 360)/360+1-v )/2
# print(gray)

height_kernel,width_kernel, = 5,5
# 2.Morphological transform
kernel = np.ones((height_kernel, width_kernel), np.uint8)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, kernel)

# 3. Image enhancement
smallest = np.amin(blackhat)
biggest = np.amax(blackhat)
blackhat = 255 * (blackhat-smallest)/(biggest-smallest)

# 4. Otsu thresholding
blackhat = blackhat.astype('uint8')
print(blackhat.dtype)
ret, otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 5. Linking discontinous line
for i in range(2, otsu.shape[0] - 2):
    for j in range(2, otsu.shape[1] - 2):
        if (otsu[i,j] == 0):
            if (otsu[i-1,j-1] + otsu[i-1,j] + otsu[i-1,j+1] + otsu[i,j-1] + otsu[i,j+1]
            + otsu[i+1,j-1] + otsu[i+1,j]+ otsu[i+1,j+1] < int((width_kernel-1)/2) * 255):
                otsu[i,j] = 0
#  6. Elimatied isolated pixel
kernel2 = np.ones((height_kernel-2, width_kernel-2), np.uint8)
opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel2)

# 7. Show on origin image
for i in range(2, otsu.shape[0] - 2):
    for j in range(2, otsu.shape[1] - 2):
        if opening[i,j] == 255:
            origin[i,j] = 255
cv2.imshow("test", origin)
cv2.waitKey(0)