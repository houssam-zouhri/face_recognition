import face_recognition
import cv2
image = face_recognition.load_image_file('./imgs/known/singe.jpg')
image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(image)

# Array of coords of each face
print(face_locations)

print(f'There are {len(face_locations)} people in this image')

cv2.imshow('houssam',image)
cv2.waitKey()
#quit()