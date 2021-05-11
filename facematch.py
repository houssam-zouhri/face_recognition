import face_recognition
import cv2

image_of_houssam = face_recognition.load_image_file('./imgs/known/houssam.jpeg')
image_of_houssam = cv2.cvtColor(image_of_houssam,cv2.COLOR_BGR2RGB)
houssam_face_encoding = face_recognition.face_encodings(image_of_houssam)[0]

unknown_image = face_recognition.load_image_file('./imgs/unknown/person4.jpeg')
unknown_image = cv2.cvtColor(unknown_image,cv2.COLOR_BGR2RGB)
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [houssam_face_encoding], unknown_face_encoding)

if results[0]:
    print('This is houssam')
else:
    print('This is not houssam')
