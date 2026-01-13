import cv2
import os
import pickle
import pandas as pd
import face_recognition
from datetime import datetime

#------------------------------------------------------------------------------------
# Function to load encodings safely
#------------------------------------------------------------------------------------
def load_encodings():
    if not os.path.exists("encodings.pkl"):
        print("\n⚠ No users found! Please add a user first.\n")
        return None, None
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

#------------------------------------------------------------------------------------
# Mark attendance only once
#------------------------------------------------------------------------------------
def mark_attendance(name):
    filename = "attendance.xlsx"

    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        df = pd.DataFrame(columns=["Name", "Time"])

    # Prevent duplicate for same session
    if name not in df["Name"].values:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[len(df)] = [name, time_now]
        df.to_excel(filename, index=False)
        print(f"\n✔ Attendance marked for {name} at {time_now}")
    else:
        print(f"\n✔ {name} attendance already recorded.")

#------------------------------------------------------------------------------------
# Function to add new user
#------------------------------------------------------------------------------------
def add_new_user():
    user = input("Enter new user name: ").strip()

    cap = cv2.VideoCapture(0)
    print("\nCapturing 20 images... Look at the camera.")

    images = []
    count = 0

    while count < 20:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capturing Images", frame)
        cv2.waitKey(50)

        images.append(frame)
        count += 1
        print(f"Captured {count}/20")

    cap.release()
    cv2.destroyAllWindows()

    print("Encoding face...")

    try:
        face_encodings = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            enc = face_recognition.face_encodings(rgb, boxes)
            if len(enc) > 0:
                face_encodings.append(enc[0])

        if len(face_encodings) == 0:
            print("❌ No face detected! Try again.")
            return

        # Save encodings
        if os.path.exists("encodings.pkl"):
            with open("encodings.pkl", "rb") as f:
                data = pickle.load(f)
            data["encodings"].extend(face_encodings)
            data["names"].extend([user] * len(face_encodings))
        else:
            data = {"encodings": face_encodings, "names": [user] * len(face_encodings)}

        with open("encodings.pkl", "wb") as f:
            pickle.dump(data, f)

        print(f"\n✔ User '{user}' added successfully!")

    except Exception as e:
        print("Error:", e)

#------------------------------------------------------------------------------------
# Mark attendance function
#------------------------------------------------------------------------------------
def mark_attendance_camera():
    known_encodings, known_names = load_encodings()
    if known_encodings is None:
        return

    cap = cv2.VideoCapture(0)
    print("\nCamera started. Looking for faces...")

    marked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        enc = face_recognition.face_encodings(rgb, boxes)

        for encoding in enc:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            if True in matches:
                index = matches.index(True)
                name = known_names[index]
                mark_attendance(name)

                marked = True
                break

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or marked:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera stopped.\n")

#------------------------------------------------------------------------------------
# Main Menu
#------------------------------------------------------------------------------------
while True:
    print("\n====== Face Recognition Attendance System ======")
    print("1. Add New User")
    print("2. Mark Attendance")
    print("3. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        add_new_user()
    elif choice == "2":
        mark_attendance_camera()
    elif choice == "3":
        print("Exiting...")
        break
    else:
        print("Invalid choice, try again.")
