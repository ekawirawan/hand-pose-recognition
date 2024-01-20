import uuid

import cv2
import firebase_admin
from firebase_admin import credentials, storage
import datetime

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

    return blob.generate_signed_url(
        response_disposition="attachment",
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow GET requests using this URL
        method="GET",
    )
