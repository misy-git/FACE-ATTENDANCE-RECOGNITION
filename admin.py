print("ADMIN PANEL")

print("1. View Attendance")
choice = input("Enter choice: ")

if choice == "1":
         with open("attendance.csv", "r") as f:
           print(f.read())