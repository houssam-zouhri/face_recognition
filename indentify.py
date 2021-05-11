import face_recognition
from PIL import Image, ImageDraw
#step 1 importation des images
image_of_houssam = face_recognition.load_image_file('./imgs/known/houssam1.jpeg')
houssam_face_encoding = face_recognition.face_encodings(image_of_houssam)[0]

image_of_abouhane = face_recognition.load_image_file('./imgs/known/mehdi.jpeg')
abouhane_face_encoding = face_recognition.face_encodings(image_of_abouhane)[0]

image_of_azzdin = face_recognition.load_image_file('./imgs/known/azzdin.jpeg')
azzdin_face_encoding = face_recognition.face_encodings(image_of_azzdin)[0]

image_of_jalol = face_recognition.load_image_file('./imgs/known/jalol.jpeg')
jalol_face_encoding = face_recognition.face_encodings(image_of_jalol)[0]

image_of_abderhmane= face_recognition.load_image_file('./imgs/known/abderhmane.jpeg')
abderhmane_face_encoding = face_recognition.face_encodings(image_of_abderhmane)[0]

image_of_abdelbassit = face_recognition.load_image_file('./imgs/known/abdelbassit.jpeg')
abdelbassit_face_encoding = face_recognition.face_encodings(image_of_abdelbassit)[0]

image_of_soukaina = face_recognition.load_image_file('./imgs/known/soukaina.jpeg')
soukaina_face_encoding = face_recognition.face_encodings(image_of_soukaina)[0]
print(len(abdelbassit_face_encoding))
print(len(soukaina_face_encoding))
#quit()
#  Create arrays of encodings and names
known_face_encodings = [
  houssam_face_encoding,
  abouhane_face_encoding,
  azzdin_face_encoding,
  jalol_face_encoding,
  soukaina_face_encoding,
  abdelbassit_face_encoding,
  abderhmane_face_encoding
]

known_face_names = [
  "houssam",
  "abouhane",
  "azzdin",
  "jalol",
  "abderhmane",
  "soukaina",
  "abdelbassit"
]

# Load test image to find faces in
test_image = face_recognition.load_image_file('./imgs/known/singe.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

  name = "Unknown Person"

  # If match
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,200))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,200))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('output/identify_group1.jpg')