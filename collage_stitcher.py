import cv2
import os
import numpy as np

# Set the maximum width and hight of each image in the collage
max_width = 3839
max_height = 705
images=[]
# Create a SIFT object
orb = cv2.ORB_create(edgeThreshold = 80,patchSize = 4,fastThreshold = 90,WTA_K=4)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

img_list=os.listdir(rf'C:\Users\Avishag\Desktop\SpinFrameRND\undercarriage\colagge_stitching\pics')
# Get a list of all the image files that you want to include in the collage
for img in img_list:
    images.append(cv2.imread(rf'.\pics\{img}'))
images=images[::-1]
    

# Calculate the number of rows and columns in the collage based on the number of images
num_images = len(images)
num_cols = 1 #number of cams
num_rows = num_images

# Create a new blank image to use as the collage
collage = np.zeros((num_rows * max_height, num_cols * max_width, 3), np.uint8)

# Paste each image into the collage

row = 0
col = 0
for image in images:
    # Resize the image to fit within the maximum width and hight
    hight,width = image.shape[:2]
    if hight > max_height or width > max_width:
        #get scale parameter for resizing
        if hight > width:
            scale = max_height / hight
        else:
            scale = max_width / width
        #resize image
        image = cv2.resize(image, (int(width * scale), int(hight * scale)))
        hight, width = image.shape[:2]
    if row==0:
         # Paste the image into the collage
            y_start = row * max_height + (max_height - hight) // 2
            y_end = y_start + hight
            x_start = col * max_width + (max_width - width) // 2
            x_end = x_start + width
            collage[y_start:y_end, x_start:x_end] = image
            # Move to the next row and column
            row += 1
            cv2.imwrite('collage.jpg', collage)
    else:
            #find matching key point and get coordinates in collage and in picture
            img1 = cv2.imread("collage.jpg")
            col_hight=y_end
            img2 = image
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            # Match the descriptors
            matches = bf.match(des1, des2)
            # Sort the matches by their distance (the lower the distance, the better the match)
            matches = sorted(matches, key=lambda x:x.distance)
            # Find the lowest keypoint in both images
            lowest_kp1 = None
            lowest_kp2 = None
            lowest_dist = float('inf')

            for match in matches:
                # Get the keypoints for the current match
                kp1_match = kp1[match.queryIdx]
                kp2_match = kp2[match.trainIdx]

                # Check if this is the lowest keypoint so far
                if (kp1_match.pt[1]-x_end < lowest_dist):
                    lowest_dist = kp1_match.pt[1]
                    lowest_kp1 = kp1_match
                    lowest_kp2 = kp2_match

            print(lowest_kp1.pt[0]) 
            print(lowest_kp1.pt[1])
            print(lowest_kp2.pt[0]) 
            print(lowest_kp2.pt[1])          
            #crop pic from kp up (leave bottom part)
            # cv2.imshow('image',image)
            image=image[int(lowest_kp2.pt[1]):,:]
            # cv2.imshow('image2',image)
            # cv2.imshow('image3',img1)
            # cv2.waitKey(99)
            hight, width = image.shape[:2]
            #add cropped pic to collage
            y_start =col_hight 
            y_end = y_start + hight
            x_start = col * max_width + (max_width - width) // 2
            x_end = x_start + width
            collage[y_start:y_end, x_start:x_end] = image
            # Move to the next row and column
            row += 1
            # if col == num_cols:
            #     col = 0
            #     row += 1
            cv2.imwrite('collage.jpg', collage)

# Save the collage
fcol=collage[:y_end,:x_end]
cv2.imwrite('collage.jpg', fcol)
