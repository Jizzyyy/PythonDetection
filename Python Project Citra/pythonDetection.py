import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from rembg import remove
from PIL import Image
import joblib
from scipy.stats import entropy
import pyrebase
import datetime
import firebase_admin
from firebase_admin import credentials, db
import time

cred = credentials.Certificate("serviceAccountKeyy.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://dataset-projectt-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Inisialisasi konfigurasi Firebase
config = {
    "apiKey": "AIzaSyA3jgPQYs6WAI_12EF6ANxhglzS4OR4HuM",
    "authDomain": "dataset-projectt.firebaseapp.com",
    "databaseURL": "https://dataset-projectt-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "projectId": "dataset-projectt",
    "storageBucket": "dataset-projectt.appspot.com",
    "messagingSenderId": "784959915731",
    "appId": "1:784959915731:web:9a11f681ed22eea51af626",
    "measurementId": "G-K7WT035DDZ",
    "serviceAccount": "serviceAccountKeyy.json"
}

firebase = pyrebase.initialize_app(config)
refdata = firebase.database()
storage = firebase.storage()
refsensor = db.reference('sensor/pir')

# Download gambar dari Firebase
storage.download("temp_photo/gambar.jpg", "sebelum_diolah/temp_photo.jpg")

# Load the trained model
model = joblib.load('trained_model.joblib')


def handle_change_data(event):
    data = event.data
    if data == 'Ada':
        print('                            ')
        print('---  Gerakan Terdeteksi  ---')
        print('-- Gambar Sedang Di Kelola --')
        print('. . . . . . . . . . . . . . .')
        try:
            extract_texture_features()
            execution()
            execution2()
        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
    else:
        # refdata.update({
        #     "ada_tikus": 0
        # })
        print('Tidak Terdeteksi Gerakan')
        time.sleep(5)

# Function to extract texture features using GLCM
def extract_texture_features():
    try:
        test_image_path = 'sebelum_diolah/temp_photo.jpg'
        test_image = cv2.imread(test_image_path)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)

        contrast = graycoprops(glcm, prop='contrast')[0, 0]
        correlation = graycoprops(glcm, prop='correlation')[0, 0]
        energy = graycoprops(glcm, prop='energy')[0, 0]
        homogeneity = graycoprops(glcm, prop='homogeneity')[0, 0]

        avg_color = np.mean(test_image, axis=(0, 1))

        return [avg_color[0], avg_color[1], avg_color[2], contrast, correlation, energy, homogeneity]
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")


def execution():
    try:
        # Load the test image membaca gambar
        test_image_path = 'sebelum_diolah/temp_photo.jpg'
        test_image = cv2.imread(test_image_path)
        pil_test_image = Image.fromarray(
            cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        removed_bg_test_image = remove(pil_test_image).convert("RGB")
        removed_bg_test_image = np.array(removed_bg_test_image)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")


def execution2():
    try:
        test_image_path = 'sebelum_diolah/temp_photo.jpg'
        test_image = cv2.cvtColor(cv2.imread(
            test_image_path), cv2.COLOR_BGR2RGB)
        removed_bg_test_image = remove(
            Image.fromarray(test_image)).convert("RGB")
        removed_bg_test_image = np.array(removed_bg_test_image)

        # Convert the original image to grayscale
        gray = cv2.cvtColor(removed_bg_test_image, cv2.COLOR_BGR2GRAY)
        # Thresholding to separate object from background
        _, thresholded = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        # Pengecekan apakah terdapat tikus yang terdeteksi
        if not contours:
            print("Tidak ada tikus yang terdeteksi.")
        else:
            for contour in contours:
                # Calculate the area of the contour
                area = cv2.contourArea(contour)

                # Set a minimum area threshold to exclude small objects (adjust as needed)
                min_area_threshold = 1000

                if area > min_area_threshold:
                    # Create a mask for the detected object
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], 0, 255, -1)

                    # Apply the mask to the grayscale image
                    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

                    # Calculate GLCM (Gray-Level Co-occurrence Matrix) for the detected object
                    glcm = graycomatrix(masked_gray, [1], [
                                        0], symmetric=True, normed=True)

                    # Calculate GLCM properties for the detected object
                    contrast = graycoprops(glcm, 'contrast')[0, 0]
                    correlation = graycoprops(glcm, 'correlation')[0, 0]
                    energy = graycoprops(glcm, 'energy')[0, 0]
                    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

                    # Calculate Average R, Average G, and Average B for the detected object
                    masked_color = cv2.bitwise_and(
                        removed_bg_test_image, removed_bg_test_image, mask=mask)
                    avg_color = np.mean(masked_color, axis=(0, 1))

                    # Additional features extraction
                    hsv = cv2.cvtColor(masked_color, cv2.COLOR_BGR2HSV)
                    avg_hsv = np.mean(hsv, axis=(0, 1))

                    # Calculate entropy
                    ent = entropy(masked_gray.flatten())

                    # Find contours
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Calculate eccentricity
                        ellipse = cv2.fitEllipse(contours[0])
                        eccentricity = np.sqrt(
                            1 - (min(ellipse[1]) / max(ellipse[1])) ** 2)

                        # Predict the category of the detected object using the trained model
                        detected_object_features = [avg_color[0], avg_color[1], avg_color[2], contrast, correlation, energy,
                                                    homogeneity, avg_hsv[0], avg_hsv[1], avg_hsv[2], ent]

                        # Print the number of features in detected_object_features
                        count_deteksi_tikus = len(detected_object_features)
                        print(
                            f"Jumlah fitur pada detected_object_features: {count_deteksi_tikus}")

                        # Check if the number of features matches the model
                        if count_deteksi_tikus == count_deteksi_tikus:
                            predicted_category = model.predict(
                                [detected_object_features])
                            detected_objects.append({
                                'contour': contour,
                                'category': predicted_category[0],
                                'features': detected_object_features
                            })

                            # Tambahkan data ke Realtime Database Firebase
                        else:
                            print(
                                "Jumlah fitur pada detected_object_features tidak sesuai dengan model.")
                    else:
                        # Default values when no contours are found
                        avg_color = np.array([0, 0, 0])
                        avg_hsv = np.array([0, 0, 0])
                        eccentricity = 0.0
                        ent = 0.0

        # Count the number of 'ada tikus' and 'tidak adatikus' eggs
        count_deteksi_tikus = sum(
            1 for obj in detected_objects if obj['category'] == 'ada tikus')
        count_tidakadatikus = sum(
            1 for obj in detected_objects if obj['category'] == 'tidak ada tikus')

        print(f"Jumlah ada tikus: {count_deteksi_tikus}")
        print(f"Tidak ada tikus: {count_tidakadatikus}")

        # Check if there is a detected mouse
        if count_deteksi_tikus > 0:
            # Draw bounding boxes around detected objects on the original image
            for obj in detected_objects:
                (x, y, w, h) = cv2.boundingRect(obj['contour'])
                cv2.rectangle(removed_bg_test_image, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(removed_bg_test_image, f"{obj['category']}", (
                    x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Buat nama gambar sesuai dengan nomor urut
            now = datetime.datetime.now()
            # Ambil nilai terakhir nomor urut deteksi_tikus dari database
            last_deteksi_tikus_index = refdata.child(
                "last_deteksi").get().val()
            if last_deteksi_tikus_index is None:
                last_deteksi_tikus_index = 0
            else:
                last_deteksi_tikus_index = int(last_deteksi_tikus_index)

            # Tambahkan 1 ke nomor urut untuk deteksi_tikus saat ini
            current_deteksi_tikus_index = last_deteksi_tikus_index + 1

            # Update nilai terakhir nomor urut deteksi_tikus di database
            refdata.child("last_deteksi").set(current_deteksi_tikus_index)

            # Buat nama deteksi_tikus sesuai dengan nomor urut
            deteksi_tikus_name = f"{current_deteksi_tikus_index:02d}"
            image_name = f"tikus_terdeteksi_{current_deteksi_tikus_index:02d}.jpg"

            # Tambahkan data ke Realtime Database Firebase
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            refdata.update({"ada_tikus": 1})

            # Upload data to Firebase
            for obj in detected_objects:
                refdata.child("deteksi_tikus").child(deteksi_tikus_name).set({
                    "nama_gambar": image_name,
                    "tanggal": now.strftime("%Y-%m-%d"),
                    "jam": now.strftime("%H:%M:%S")
                })

                cv2.imwrite(f'sesudah_diolah/{image_name}',
                            removed_bg_test_image)

                storage.child(
                    f"data/image/{image_name}").put(f"sesudah_diolah/{image_name}")

            print("Data dan gambar terdeteksi tikus berhasil diunggah ke Firebase.")
        else:
            # refdata.update({
            #     "ada_tikus": 0
            # })
            # db.set({"ada_tikus": 0})
            print("Tidak ada tikus yang terdeteksi.")

        # cv2.imshow('Detected Objects', removed_bg_test_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")


refsensor.listen(handle_change_data)

while True:
    pass
