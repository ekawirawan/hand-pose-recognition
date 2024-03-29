import threading
from typing import Union

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import (
    VideoProcessorBase,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

from helpers.filter_image import handEmote, read_emote_pose, read_face_haarcascade
from helpers.turn import get_ice_servers
from helpers.upload_image import upload_image

PATH_TO_MODEL = "./custom_model_lite/detect.tflite"
PATH_TO_LABELS = "./custom_model_lite/labelmap.txt"


@st.cache_resource
def load_tf_lite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=PATH_TO_MODEL)
        interpreter.allocate_tensors()

        return interpreter
    except ValueError as ve:
        print("Error loading the TensorFlow Lite model:", ve)
        exit()


@st.cache_resource
def load_labels():
    with open(PATH_TO_LABELS, "r") as f:
        labels = [line.strip() for line in f.readlines()]

        return labels


def get_model_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    float_input = input_details[0]["dtype"] == np.float32

    return input_details, output_details, height, width, float_input


face_cas = read_face_haarcascade()
posev_emote, thumb_emote, metal_emote = read_emote_pose()


class VideoTransformer(VideoTransformerBase):
    frame_lock: threading.Lock
    out_image: Union[np.ndarray, None]

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.out_image = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        interpreter = load_tf_lite_model()

        labels = load_labels()

        input_mean = 127.5
        input_std = 127.5

        # get model details
        input_details, output_details, height, width, float_input = get_model_details(
            interpreter=interpreter
        )

        out_image = frame.to_ndarray(format="bgr24")

        imH, imW, _ = out_image.shape

        image_resized = cv2.resize(out_image, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]["index"])[0]
        classes = interpreter.get_tensor(output_details[3]["index"])[0]
        scores = interpreter.get_tensor(output_details[0]["index"])[0]

        for i in range(len(scores)):
            if (scores[i] > 0.5) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(out_image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])]

                label = "%s: %d%%" % (
                    object_name,
                    int(scores[i] * 100),
                )

                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                label_ymin = max(ymin, labelSize[1] + 10)

                cv2.rectangle(
                    out_image,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )

                cv2.putText(
                    out_image,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

                out_image = handEmote(
                    object_name,
                    out_image,
                    face_cas,
                    posev_emote,
                    thumb_emote,
                    metal_emote,
                )

                with self.frame_lock:
                    self.out_image = out_image

        return out_image


def realtime_video_detection():
    info = st.empty()
    info.markdown("First, click on :blue['START'] to use webcam")
    ctx = webrtc_streamer(
        key="object detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_transformer:
        info.markdown("Click on :blue['SNAPSHOT'] to take a picture")
        snap = st.button("SNAPSHOT")
        if snap:
            with ctx.video_transformer.frame_lock:
                out_image = ctx.video_transformer.out_image

            if out_image is not None:
                st.write("Result:")
                st.image(out_image, channels="BGR")
                public_url = upload_image(out_image)
                st.markdown(
                    f"<a href='{public_url}' download='hand-pose-recognition.jpg' style='display: inline-flex; -webkit-box-align: center; align-items: center; -webkit-box-pack: center; justify-content: center; font-weight: 400; padding: 0.25rem 0.75rem; border-radius: 0.5rem; min-height: 38.4px; margin: 0px; line-height: 1.6; color: inherit; width: auto; user-select: none; background-color: rgb(19, 23, 32); border: 1px solid rgba(250, 250, 250, 0.2); text-decoration: none;'>Download Photo &#10515;</a>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No frames available yet.")
