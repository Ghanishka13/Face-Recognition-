import cv2
import face_recognition
import pickle
import os

name = input("Enter the name of the user: ")

# Load existing encodings
if os.path.exists("encodings.pkl"):
    data = pickle.load(open("encodings.pkl", "rb"))
    known_encodings = data["encodings"]
    known_names = data["names"]
else:
    known_encodings = []
    known_names = []

cam = cv2.VideoCapture(0)
print("Camera ON - Press 'c' to capture image...")

captured_frame = None

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb)

    # Draw rectangles around detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Add User - Press C to Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if len(face_locations) == 0:
            print("❌ No face detected! Try again.")
            continue
        captured_frame = frame
        break

cam.release()
cv2.destroyAllWindows()

if captured_frame is None:
    print("No image captured.")
    exit()

# Convert BGR to RGB for encoding
rgb = captured_frame[:, :, ::-1]

# Encode detected face(s)
face_locations = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, face_locations)

if len(encodings) == 0:
    print("❌ Could not encode face. Try again.")
    exit()

# Add encoding to file
known_encodings.append(encodings[0])
known_names.append(name)

with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"✅ User '{name}' added successfully!")
