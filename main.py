import subprocess
import os

def menu():
    print("\n===== FACE RECOGNITION SYSTEM =====")
    print("1. Add New User")
    print("2. Mark Attendance")
    print("3. Exit")
    choice = input("Enter your choice: ")
    return choice

while True:
    ch = menu()

    if ch == "1":
        subprocess.run(["python", "add_user.py"])

    elif ch == "2":
        subprocess.run(["python", "attendance.py"])

    elif ch == "3":
        print("Exiting system...")
        break

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
