import uuid
import cv2

import firebase_admin
from firebase_admin import credentials, storage

PATH_TO_SERVICE_ACCOUNT = "venv/serviceAccount.json"


if not firebase_admin._apps:
    cred = credentials.Certificate(PATH_TO_SERVICE_ACCOUNT)
    app = firebase_admin.initialize_app(
        cred, {"storageBucket": "hand-pose-recognition.appspot.com"}
    )


def upload_image(img_array):
    _, img_data = cv2.imencode(".jpg", img_array)
    img_byte_array = img_data.tobytes()
    bucket = storage.bucket()

    random_filename = str(uuid.uuid4())

    destination_path = f"images/{random_filename}.jpg"
    blob = bucket.blob(destination_path)

    blob.upload_from_string(img_byte_array, content_type="image/jpeg")

    return destination_path


def download_image(file_name):
    bucket = storage.bucket()

    blob = bucket.blob(file_name)

    img_data = blob.download_as_bytes()

    return img_data


# file_name_uploaded = upload_image("../../test.jpg")

# local_download_path = (
#     "../../fromfirebase.jpg"  # path lokal untuk menyimpan file yg diunduh
# )

# download_image(file_name_uploaded, local_download_path)
