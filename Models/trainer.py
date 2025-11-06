# import cv2
# import os
# import numpy as np
# from PIL import Image

# # Paths
# DATASET_DIR = "dataset"
# TRAINER_DIR = "trainer"

# # Create trainer directory if it doesn't exist
# if not os.path.exists(TRAINER_DIR):
#     os.makedirs(TRAINER_DIR)

# # Initialize recognizer and face detector
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def getImagesAndLabels(path):
#     faceSamples = []
#     Ids = []
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    
#     for imagePath in imagePaths:
#         # Only process image files (e.g. jpg, png)
#         if not imagePath.lower().endswith(('.jpg', '.jpeg', '.png')):
#             continue

#         # Convert image to grayscale
#         PIL_img = Image.open(imagePath).convert('L')
#         img_numpy = np.array(PIL_img, 'uint8')

#         # Extract ID from filename (e.g. user.1.1.jpg → Id = 1)
#         try:
#             Id = int(os.path.split(imagePath)[-1].split(".")[1])
#         except:
#             print(f"⚠️ Skipping {imagePath} — invalid file name format.")
#             continue

#         # Detect faces
#         faces = detector.detectMultiScale(img_numpy)
#         for (x, y, w, h) in faces:
#             faceSamples.append(img_numpy[y:y + h, x:x + w])
#             Ids.append(Id)

#     return faceSamples, Ids


# print("🔹 Training faces... Please wait.")
# faces, Ids = getImagesAndLabels(DATASET_DIR)

# if len(faces) == 0:
#     print("❌ No faces found. Make sure dataset is prepared correctly.")
# else:
#     recognizer.train(faces, np.array(Ids))
#     recognizer.save(f"{TRAINER_DIR}/trainer.yml")
#     print(f"✅ {len(np.unique(Ids))} Faces Trained. Model saved at {TRAINER_DIR}/trainer.yml")






# # import cv2
# # import os
# # import numpy as np
# # from PIL import Image

# # # ---------- PATHS ----------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
# # TRAINER_PATH = os.path.join(BASE_DIR, 'trainer.yml')
# # CASCADE_PATH = os.path.join(BASE_DIR, '../haarcascade/haarcascade_frontalface_default.xml')

# # # ---------- FUNCTION TO GET IMAGES AND LABELS ----------
# # def getImagesAndLabels(path):
# #     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
# #     faces = []
# #     Ids = []
# #     detector = cv2.CascadeClassifier(CASCADE_PATH)

# #     for imagePath in imagePaths:
# #         pilImage = Image.open(imagePath).convert('L')
# #         imageNp = np.array(pilImage, 'uint8')
# #         Id = int(os.path.split(imagePath)[-1].split('.')[1])
# #         faces_detected = detector.detectMultiScale(imageNp)
# #         for (x, y, w, h) in faces_detected:
# #             faces.append(imageNp[y:y+h, x:x+w])
# #             Ids.append(Id)
# #     return faces, Ids

# # # ---------- MAIN TRAINING EXECUTION ----------
# # if __name__ == '__main__':
# #     print("🔹 Training faces...")
# #     recognizer = cv2.face.LBPHFaceRecognizer_create()
# #     faces, Ids = getImagesAndLabels(DATASET_DIR)
# #     recognizer.train(faces, np.array(Ids))
# #     recognizer.save(TRAINER_PATH)
# #     print("✅ Training complete. Model saved at:", TRAINER_PATH)


import os
import json
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Models/
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
NAMES_FILE = os.path.join(TRAINER_DIR, "names.txt")
ID_NAME_MAP = os.path.join(DATASET_DIR, "id_name_map.json")

os.makedirs(TRAINER_DIR, exist_ok=True)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_id_name_map(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_images_and_labels(dataset_path):
    image_paths = []
    for fname in os.listdir(dataset_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(dataset_path, fname))
    image_paths.sort()

    face_samples = []
    ids = []
    for imagePath in image_paths:
        fname = os.path.basename(imagePath)
        parts = fname.split(".")
        # expected pattern: User.<ID>.<sample>.<ext>  e.g. User.1.1.jpg
        if len(parts) < 3:
            # skip unexpected file names
            continue
        try:
            Id = int(parts[1])
        except:
            continue

        try:
            pil_img = Image.open(imagePath).convert('L')
        except Exception as e:
            print(f"⚠️ Could not open {imagePath}: {e}")
            continue

        img_numpy = np.array(pil_img, 'uint8')
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(Id)
    return face_samples, ids

def main():
    id_name_map = load_id_name_map(ID_NAME_MAP)
    if not id_name_map:
        print(f"⚠️ id_name_map.json not found or empty at {ID_NAME_MAP}. Create or update it using dataset_creator.")
        # still continue training if files exist and numeric ids are present

    print("🔹 Preparing training data...")
    faces, ids = get_images_and_labels(DATASET_DIR)

    if len(faces) == 0:
        print("❌ No faces found in dataset. Ensure files like User.<ID>.<num>.jpg exist in dataset/")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)
    print(f"✅ Training complete. Model saved at {TRAINER_FILE}")

    # Save names mapping to a simple names.txt for quick loading by detector
    # If id_name_map.json exists, use that; otherwise build names from unique ids
    if id_name_map:
        with open(NAMES_FILE, "w", encoding="utf-8") as f:
            for id_str, name in id_name_map.items():
                f.write(f"{int(id_str)},{name}\n")
    else:
        # build from ids discovered
        uniq = sorted(set(ids))
        with open(NAMES_FILE, "w", encoding="utf-8") as f:
            for uid in uniq:
                f.write(f"{uid},ID_{uid}\n")
    print(f"✅ Names file saved at {NAMES_FILE}")

if __name__ == "__main__":
    main()
