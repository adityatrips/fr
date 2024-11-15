// init-mongo.js
db = db.getSiblingDB("lens_face_det"); // Switch to the 'lens_face_det' database
db.createCollection("face_features"); // Create the 'face_features' collection
