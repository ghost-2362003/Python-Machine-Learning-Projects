import cv2 as cv
img = cv.imread("C:/Users/shubh/Desktop/profile-pic.jpg")

#show the image
img = cv.resize(img, (512, 512))
cv.imshow("Display window", img)
print(img.shape)
cv.waitKey(0)