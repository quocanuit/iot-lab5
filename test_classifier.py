import os
import numpy as np
import tensorflow as tf
import pickle
import facenet
import cv2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.align import detect_face

# -------------------------------
# Configuration
# -------------------------------
TEST_DIR              = "Dataset/test"
FACENET_MODEL_PATH    = "Models/20180402-114759.pb"
CLASSIFIER_PATH       = "Models/facemodel.pkl"
INPUT_IMAGE_SIZE      = 160
MINSIZE               = 20
THRESHOLD             = [0.6, 0.7, 0.7]
FACTOR                = 0.709

# -------------------------------
# Load classifier
# -------------------------------
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print(f"Loaded classifier model from file \"{CLASSIFIER_PATH}\"")

# -------------------------------
# Setup TensorFlow and MTCNN
# -------------------------------
with tf.Graph().as_default():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        facenet.load_model(FACENET_MODEL_PATH)

        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        # -------------------------------
        # Load test data
        # -------------------------------
        image_paths = []
        labels = []
        for label in os.listdir(TEST_DIR):
            person_dir = os.path.join(TEST_DIR, label)
            if not os.path.isdir(person_dir):
                continue
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(person_dir, img_name))
                    labels.append(label)

        emb_array = []
        true_labels = []

        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                continue

            # Detect faces
            bounding_boxes, _ = detect_face.detect_face(
                img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            if bounding_boxes.shape[0] == 0:
                # print(f"No face detected in {path}")
                continue

            # Use first detected face
            box = bounding_boxes[0, :4].astype(int)
            margin = 10
            x1 = max(box[0] - margin, 0)
            y1 = max(box[1] - margin, 0)
            x2 = min(box[2] + margin, img.shape[1])
            y2 = min(box[3] + margin, img.shape[0])
            cropped = img[y1:y2, x1:x2]
            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
            prewhitened = facenet.prewhiten(scaled)
            prewhitened = prewhitened.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

            # Get embedding
            feed_dict = {images_placeholder: prewhitened, phase_train_placeholder: False}
            emb = sess.run(embeddings_tensor, feed_dict=feed_dict)
            emb_array.append(emb[0])
            true_labels.append(labels[i])

        emb_array = np.array(emb_array)
        true_labels = np.array(true_labels)

        # -------------------------------
        # Predict
        # -------------------------------
        preds = model.predict(emb_array)
        probs = model.predict_proba(emb_array)

        # acc = accuracy_score(true_labels, preds)
        # print(f"\nAccuracy: {acc:.3f}\n")

        # -------------------------------
        # Print per-class probabilities
        # -------------------------------
        class_prob = {name: [] for name in class_names}
        for pred, prob, label in zip(preds, probs, true_labels):
            if label in class_prob:
                class_prob[label].append(np.max(prob))

        for name in class_names:
            if class_prob[name]:
                avg = np.mean(class_prob[name])
                print(f"{name}: {avg:.3f}")
