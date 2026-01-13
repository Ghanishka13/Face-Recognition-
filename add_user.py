import face_recognition           # Face recognition library library function provode karta hai face ko encode karne me 
import cv2                   # OpenCV library computer vision tasks ke liye use hoti hai , camera acess karne ke liye,image to frame me convert karne ke liye and color space conversion ke liye
import pickle            # Python object ko byte stream me convert karne ke liye use hota hai , jisse hum apne face encodings ko file me save kar saken
import os                # Operating system ke sath interact karne ke liye use hota hai , file existence check karne ke liye

name = input("Enter the name of the user: ")       # User ka naam input lete hai

if os.path.exists("encodings.pickle"):           # Check karte hai ki encodings.pickle file exist karti hai ya nahi
    data = pickle.load(open("encodings.pickle", "rb"))  # Agar file exist karti hai to usme se data load karte hai and data isme stored rhta hai (rb)read-binary mode me
    known_encodings = data["encodings"]  # Existing encodings ko known_encodings variable me store karte hai
    known_names = data["names"]             # Existing names ko known_names variable me store karte hai
else:
    known_encodings = [] # Agar file exist nahi karti to empty list initialize karte hai
    known_names = []             # Agar file exist nahi karti to empty list initialize karte hai

cam = cv2.VideoCapture(0)    # Camera ko access karne ke liye OpenCV ka VideoCapture function use karte hai , 0 default camera ko refer karta hai
print("Camera ON - Press 'c' to capture image...")    # Camera ON message show karte hai and user ko press c key pehle tak wait karna hai

captured_frame = None    # Captured frame ko None initialize karte hai

while True:   # Infinite loop me camera se frames read karte hai
    ret, frame = cam.read()  # Camera se frame read karte hai
    cv2.imshow("Add User - Press C to Capture", frame) # Frame ko imshow function se show karte hai
    if cv2.waitKey(1) & 0xFF == ord('c'):  # Agar user 'c' key press karta hai to captured_frame variable me current frame store karte hai and loop break hoga
        captured_frame = frame                 # Captured frame ko store karte hai
        break

cam.release()  # Camera release karte hai
cv2.destroyAllWindows() # Sabhi OpenCV windows close karte hai

if captured_frame is None:
    print("❌ No image captured.")
    exit()

rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)  # Captured frame ko BGR se RGB color space me convert karte hai for ML

# STEP 1: Face detection
face_locations = face_recognition.face_locations(rgb, model="cnn") # Face locations detect karte hai using CNN model

if len(face_locations) == 0:   # Agar koi face detect nahi hota to user ko message show karte hai
    face_locations = face_recognition.face_locations(rgb, model="hog") # Agar CNN model se face detect nahi hota to HOG (Histogram of Oriented Gradients)model use karte hai

if len(face_locations) == 0:
    print("❌ No face detected. Improve lighting or distance.")
    exit()

# 🛡️ STEP 2: Face landmark extraction (SAFE)
try:         
    encodings = face_recognition.face_encodings(rgb, face_locations, num_jitters=1)  # Face encodings nikalte hai using face locations
except:
    print("❌ Landmark error. Try facing the camera directly.")
    exit()

if len(encodings) == 0:
    print("❌ Face detected but could not encode. Try again.")
    exit()

# Save encoding
known_encodings.append(encodings[0])                   # New encoding ko known_encodings list me add karte hai
known_names.append(name)

with open("encodings.pickle", "wb") as f:        # Updated encodings and names ko encodings.pickle file me write karte hai (wb) write-binary mode me
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)      # Pickle module se dump function use karte hai

print(f"✅ User '{name}' added successfully!")
