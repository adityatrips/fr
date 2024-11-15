import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient
from env import MONGO
from uuid import uuid4

client = MongoClient(MONGO)


class FaceRecognition:
    COSINE_THRESHOLD = 0.65

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.dictionary = {}
        self.face_detector = None
        self.face_recognizer = None
        self._init_models()
        self.load_images()

    def _init_models(self):
        detection_weights = os.path.join(
            os.path.abspath(os.curdir),
            self.data_dir,
            "models",
            "face_detection_yunet_2022mar.onnx",
        )
        self.face_detector = cv2.FaceDetectorYN_create(detection_weights, "", (0, 0))
        self.face_detector.setScoreThreshold(0.87)

        recognition_weights = os.path.join(
            self.data_dir, "models", "face_recognizer_fast.onnx"
        )
        self.face_recognizer = cv2.FaceRecognizerSF_create(recognition_weights, "")

    def load_images(self):
        # for image in tqdm(glob.glob("data/images/*")):
        #     features, _ = self._recognize_face(image)
        #     if features:
        #         for feature in features:
        #             client["lens_face_det"]["face_features"].insert_one(
        #                 {"uuid": str(uuid4()), "feature": feature.tolist()}
        # )

        documents = client["lens_face_det"]["face_features"].find({})
        for document in documents:
            self.dictionary[document["uuid"]] = np.array(
                document["feature"], dtype=np.float32
            )

    def add_user(self, img_bytes: bytes, identifier: str):
        uuid = str(uuid4())
        im_path = str(
            os.path.join(
                os.path.abspath(os.path.curdir), "data", "images", f"{uuid}.jpg"
            )
        )
        print(im_path)
        with open(im_path, "wb") as f:
            f.write(img_bytes)
        print(im_path)

        print("Before imread")
        img = cv2.imread(im_path)
        print("After imread")
        h, w = img.shape[:2]

        new_w = 600
        new_h = int(h * new_w / w)
        img = cv2.resize(img, (new_w, new_h))
        # UUID generation
        # Feature extraction
        features, _ = self._recognize_face(im_path)
        # MongoDb store (features, and uuid)
        self.dictionary[uuid] = features[0]
        client["lens_face_det"]["face_features"].insert_one(
            {"uuid": uuid, "feature": features[0].tolist(), "identifier": identifier}
        )
        # Return uuid, and image_path
        return uuid, img

    # def populate_db(self):
    #     for im in tqdm(glob.glob("data/images/*")):
    #         img = cv2.imread(im)
    #         uuid = str(uuid4())
    #         h, w = img.shape[:2]
    #         new_w = 600
    #         new_h = int(h * new_w / w)
    #         img = cv2.resize(img, (new_w, new_h))
    #         # UUID generation
    #         # Feature extraction
    #         features, _ = self._recognize_face(img)
    #         # MongoDb store (features, and uuid)
    #         try:
    #             client["lens_face_det"]["face_features"].insert_one(
    #                 {"uuid": uuid, "feature": features[0].tolist(), "identifier": uuid}
    #             )
    #         except Exception as e:
    #             print(e)
    #             continue

    def _recognize_face(self, image):
        image = cv2.imread(image)

        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.shape[0] > 1000:
            image = cv2.resize(
                image,
                (0, 0),
                fx=0.5,
                fy=0.5,
            )

        height, width, _ = image.shape
        self.face_detector.setInputSize((width, height))

        try:
            _, faces = self.face_detector.detect(image)

            features = []
            for face in faces:
                aligned_face = self.face_recognizer.alignCrop(image, face)
                feat = self.face_recognizer.feature(aligned_face)
                features.append(feat)

            return features, faces
        except Exception as e:
            return None, None

    def match_face(self, feature1):
        max_score = 0.0
        sim_user_id = ""

        print

        for user_id, feature2 in self.dictionary.items():
            print(type(feature1))
            print()
            print(type(feature2))

            score = self.face_recognizer.match(
                feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > max_score:
                max_score = score
                sim_user_id = user_id

        return (
            (
                sim_user_id,
                max_score,
            )
            if max_score >= self.COSINE_THRESHOLD
            else (None, 0.0)
        )

    def recognize_input_image(self, im_path):
        features, faces = self._recognize_face(im_path)

        for feature in features:
            user_id, score = self.match_face(feature)
            if user_id:
                return user_id, score

        return None, 0.0


if __name__ == "__main__":
    fr = FaceRecognition()

    fr.populate_db()
