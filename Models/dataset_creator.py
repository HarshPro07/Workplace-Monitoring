import cv2
import os
import sqlite3
import json

# ---------- DATABASE SETUP ----------
conn = sqlite3.connect("FaceBase.db")
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS Peoples
    (ID INTEGER PRIMARY KEY, Name TEXT, Age INTEGER, Gender TEXT)''')
conn.commit()

# ---------- FOLDER & FILE SETUP ----------
DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

ID_NAME_MAP_PATH = os.path.join(DATASET_DIR, "id_name_map.json")

if os.path.exists(ID_NAME_MAP_PATH):
    with open(ID_NAME_MAP_PATH, "r") as f:
        id_name_map = json.load(f)
else:
    id_name_map = {}

# ---------- INSERT OR UPDATE ----------
def insert_or_update(Id, Name, Age, Gender):
    cursor = conn.execute("SELECT * FROM Peoples WHERE ID=?", (Id,))
    if cursor.fetchone():
        conn.execute("UPDATE Peoples SET Name=?, Age=?, Gender=? WHERE ID=?", (Name, Age, Gender, Id))
    else:
        conn.execute("INSERT INTO Peoples (ID, Name, Age, Gender) VALUES (?, ?, ?, ?)", (Id, Name, Age, Gender))
    conn.commit()

# ---------- INPUT DETAILS ----------
Id = input('Enter Employee ID: ')
Name = input('Enter Name: ')
Age = input('Enter Age: ')
Gender = input('Enter Gender: ')

insert_or_update(Id, Name, Age, Gender)
id_name_map[str(Id)] = Name

with open(ID_NAME_MAP_PATH, "w") as f:
    json.dump(id_name_map, f, indent=4)
print(f"✅ Updated ID-Name Map with {Id}: {Name}")

# ---------- CAPTURE FACES ----------
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
sampleNum = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"{DATASET_DIR}/User.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"{Name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow('Capturing Faces', img)

    if cv2.waitKey(100) & 0xFF == 27:
        break
    elif sampleNum >= 30:
        break

cam.release()
cv2.destroyAllWindows()
conn.close()
print("✅ Dataset created successfully!")
