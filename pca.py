# import modules
import numpy as np
import cv2
# defining image width and height
image_width = 320
image_length = 243
# used fro flattening of image
total_pixels = image_width * image_length
# total number of people
images = 10
# expressions of each people
variants = 5
total_images = images * variants

facevector = []
# flattening the images in the database
for i in range(1, total_images + 1):
    path = "database/" + str(i) + '.jpg'
    # read the image
    img = cv2.imread(path)
    # convert to grayscale for faster calculations
    face_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_image = face_image.reshape(total_pixels, )
    facevector.append(face_image)

face_vector = np.asarray(facevector)

face_vector = face_vector.transpose()
# find mean face-vector
# deflattening of the mean vector would give us average image
avg_face_vector = face_vector.mean(axis=1)

avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
# normalized face vectors after subtracting mean from each image vector
normalized_face_vector = face_vector - avg_face_vector
# finding covariance matrix
covariance_matrix = np.cov(np.transpose(normalized_face_vector))
# calculating eigen values and eigen values
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
# select best k eigenfaces
k = 10
k_eigen_vectors = eigen_vectors[0:k, :]
eigen_faces = k_eigen_vectors.dot(np.transpose(normalized_face_vector))
# deflattening of eigen faces gives eigenfaces images

weights = np.transpose(normalized_face_vector).dot(np.transpose(eigen_faces))


# testing of database
# number of correct responses
count = 0
positive_test_images = 50
if k == 5:
    s = 1.3
elif k == 10:
    s = 1.2
else:
    s = 1.1

# testing postive images
for i in range(1, positive_test_images + 1):
    # reading the image
    test_add = "test_data/positive/" + str(i) + ".jpg"
    gg = cv2.imread('test_data/positive/' + str(i) + '.jpg')
    test_img = cv2.imread(test_add)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    original_image = test_img
    # face detection using Haar cascade
    face_cascade = cv2.CascadeClassifier('cascade.xml')
    detected_faces = face_cascade.detectMultiScale(original_image, s, 7)
    # print(len(detected_faces))
    # if no faces found then no need to do recognition
    if len(detected_faces) == 0:
        continue

    test_img = test_img.reshape(total_pixels, 1)
    test_normalized_face_vector = test_img - avg_face_vector
    test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))
    array = np.linalg.norm(test_weight - weights, axis=1)

    # finding minimum distance
    dis = np.min(np.linalg.norm(test_weight - weights, axis=1))
    # index stores the position of image in database if found
    index = -1
    for i in range(total_images):
        if dis == array[i]:
            index = i
    # threshhold value is 1
    if dis > 1:
        index = -1
    if index != -1:
        count = count + 1
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    # color in BGR
    color = (0, 255, 0)
    thickness = 2
    # finding name of person using index value
    name = "person" + str(index//5 +1)
    x = 0
    y = 0
    w = 0
    z = 0
    max = 0
    for (a, b, c, d) in detected_faces:
        if c * d >= max:
            x = a
            y = b
            w = c
            h = d

    org = (x, y)
    # drawing rectangle with maximum area
    cv2.rectangle(gg, (x, y), (x + w, y + h), (255, 0, 0), 5)
    image = cv2.putText(gg, name, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Image', gg)
    # change the below parameter if you want slow video show
    cv2.waitKey(300)
negative_test_images = 5
# testing postive images
for i in range(1, negative_test_images + 1):
    # reading image
    test_add = "test_data/negative/" + str(i+14) + ".jpg"
    gg = cv2.imread('test_data/negative/' + str(i+14) + '.jpg')
    test_img = cv2.imread(test_add)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    original_image = test_img
    # face detection
    face_cascade = cv2.CascadeClassifier('cascade.xml')
    detected_faces = face_cascade.detectMultiScale(original_image, 1.1, 7)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 0
    y = 0
    w = 0
    z = 0
    max = 0
    for (a, b, c, d) in detected_faces:
        if c * d >= max:
            x = a
            y = b
            w = c
            h = d

    org = (x, y)
    # drawing rectangle with maximum area if found
    cv2.rectangle(gg, (x, y), (x + w, y + h), (255, 0, 0), 5)
    # putting text of not found
    image = cv2.putText(gg, "Not found", (x, y), font,
                        1, (0, 255, 0), 2, cv2.LINE_AA)


    test_img = test_img.reshape(total_pixels, 1)
    test_normalized_face_vector = test_img - avg_face_vector
    test_weight = np.transpose(test_normalized_face_vector).dot(np.transpose(eigen_faces))
    # compare test_weight with weights array and find minimum distance
    array = np.linalg.norm(test_weight - weights, axis=1)

    dis = np.min(np.linalg.norm(test_weight - weights, axis=1))
    index = -1
    for i in range(total_images):
        if dis == array[i]:
            index = i
    # threshhold value is 1
    if dis > 1:
        index = -1
    if index == -1:
        count = count + 1
    cv2.imshow('Image', gg)
    # change the parameter to slow down the output video
    cv2.waitKey(300)
accuracy = (count/(positive_test_images + negative_test_images)) * 100
print("Accuracy is", accuracy)

cv2.destroyAllWindows()
