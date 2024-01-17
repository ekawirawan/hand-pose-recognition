import cv2
import keras
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="Camera Filter",
    page_icon=":camera_with_flash:",
)


st.title("Camera Filter :sparkles:")


model = load_model("model.h5")

classes = ["posev", "thumb", "metal"]

width = 516
height = 300

picture = st.camera_input("First, take a picture...")


def put_emote(emote, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width + 1
    hat_height = int(0.50 * face_height) + 1

    try:
        emote = cv2.resize(emote, (hat_width, hat_height))
    except Exception as e:
        print(f"Error resizing emote image: {e}")

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if emote[i][j][k] < 235:
                    fc[y + i - int(-0.20 * face_height)][x + j][k] = emote[i][j][k]
    return fc


def handEmote(pose):
    face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    posevEmote = cv2.imread("assets\illustrations\hand-pose\posev.png")
    thumbEmote = cv2.imread("assets\illustrations\hand-pose\thumb.png")
    metalEmote = cv2.imread("assets\illustrations\hand-pose\metal.png")

    frame = cv2.imread(
        "WIN_20240111_23_06_04_Pro_jpg.rf.1d9eea908a8ab82dfc771fdec8ad8d61.jpg",
    )

    face_det = face_cas.detectMultiScale(frame, 1.3, 5)
    emote = ""
    for x, y, w, h in face_det:
        if pose == "posev":
            emote = posevEmote
        elif pose == "thumb":
            emote = thumbEmote
        elif pose == "metal":
            emote = metalEmote
        else:
            print("error")

        frame = put_emote(emote, frame, x, y, w, h)

    st.markdown("Filter Result")
    st.image(
        frame,
        use_column_width=True,
    )


if picture:
    with open("test.jpg", "wb") as file:
        file.write(picture.getbuffer())

if picture is not None:
    path_img = "test.jpg"
    image = Image.open(path_img)
    resized_image = image.resize((width, height))
    resized_image.save("resize.jpg")

    img = keras.preprocessing.image.load_img(
        "WIN_20240111_23_06_04_Pro_jpg.rf.1d9eea908a8ab82dfc771fdec8ad8d61.jpg",
        target_size=(height, width),
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img_array, 0)
    predictions = model.predict(img)

    bbox = predictions[1][0]
    bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]

    class_prediction_value = predictions[0][0]
    score = tf.nn.softmax(class_prediction_value)

    score = tf.math.argmax(score)
    output_class = classes[score]

    img = keras.preprocessing.image.load_img(
        "WIN_20240111_23_06_04_Pro_jpg.rf.1d9eea908a8ab82dfc771fdec8ad8d61.jpg",
        target_size=(height, width),
    )

    cv2.rectangle(
        img_array,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (96, 180, 255),
        2,
    )

    label_position = (int(bbox[0]), int(bbox[1] - 15))

    cv2.putText(
        img_array,
        output_class,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (96, 180, 255),
        2,
    )

    img_array_normalized = img_array / 255.0

    handEmote(output_class)

    st.markdown("Pose Detected")

    st.image(
        img_array_normalized,
        use_column_width=True,
    )
