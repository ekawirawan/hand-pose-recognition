import firebase_admin
from firebase_admin import credentials, storage
import uuid

PATH_TO_SERVICE_ACCOUNT = "serviceAccount.json"

cred = credentials.Certificate(PATH_TO_SERVICE_ACCOUNT)
app = firebase_admin.initialize_app(
    cred, {"storageBucket": "hand-pose-recognition.appspot.com"}
)


def upload_image(local_file_path):
    bucket = storage.bucket()

    random_filename = str(uuid.uuid4())

    destination_path = f"images/{random_filename}.jpg"
    blob = bucket.blob(destination_path)

    blob.upload_from_filename(local_file_path)

    return destination_path


def download_image(source_path, local_file_path):
    bucket = storage.bucket()
    blob = bucket.blob(source_path)

    # Download ke file lokal
    blob.download_to_filename(local_file_path)


file_name_uploaded = upload_image("../../test.jpg")

local_download_path = (
    "../../fromfirebase.jpg"  # path lokal untuk menyimpan file yg diunduh
)

download_image(file_name_uploaded, local_download_path)
