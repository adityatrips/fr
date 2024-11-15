from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from fr import FaceRecognition
from time import time
import os
from pymongo import MongoClient
from env import MONGO

client = MongoClient(MONGO)
db = client["lens_face_det"]
collection = db["face_features"]

fr = FaceRecognition(data_dir="data")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sign-in")
def sign_in_existing_user(image: bytes = File(...)):
    img_path = os.path.join(os.path.abspath(os.path.curdir), "input_image.jpg")
    try:
        st = time()
        with open(img_path, "wb") as f:
            f.write(image)
        matched_user_id, score = fr.recognize_input_image(img_path)
        print(score, matched_user_id)
        identifier = collection.find_one({"uuid": matched_user_id})
        en = time()
        if matched_user_id is not None:
            return {
                "success": True,
                "uuid": matched_user_id,
                "confidence": float(f"{score:.2f}"),
                "time_s": float(f"{en-st:.2f}"),
                "message": identifier["identifier"],
            }
        else:
            return {
                "success": False,
                "uuid": None,
                "confidence": float(f"{score:.2f}"),
                "time_s": float(f"{en-st:.2f}"),
                "message": "User not recognized!",
            }
    except Exception as e:
        print(e)
        return {
            "success": False,
            "uuid": None,
            "confidence": 0,
            "time_s": 0,
            "message": "Something went wrong!",
        }


@app.post("/sign-up")
def create_new_user(image: bytes = File(...), identifier: str = ""):
    try:
        im_path = os.path.join(os.path.abspath(os.path.curdir), "input_image.jpg")
        with open(im_path, "wb") as f:
            f.write(image)
        matched_user_id, score = fr.recognize_input_image(im_path)
        if matched_user_id is not None:
            return {
                "success": True,
                "uuid": matched_user_id,
                "message": "User already exists! Please login.",
            }
        else:
            user_id, img = fr.add_user(image, identifier)
            return {
                "success": True,
                "uuid": user_id,
                "message": "User not recognized!",
            }
    except Exception as e:
        print(e)
        return {
            "success": False,
            "uuid": None,
            "message": "Something went wrong!",
        }
