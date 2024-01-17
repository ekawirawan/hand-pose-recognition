import cv2
import numpy as np
import tensorflow as tf

import streamlit as st

PATH_TO_MODEL = "custom_model_lite/detect.tflite"
PATH_TO_LABELS = "custom_model_lite/labelmap.txt"


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


def detect_images(image_path, min_conf=0.5):
    interpreter = load_tf_lite_model()
    labels = load_labels()

    # get model details
    input_details, output_details, height, width, float_input = get_model_details(
        interpreter=interpreter
    )

    input_mean = 127.5
    input_std = 127.5

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
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
        if (scores[i] > min_conf) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

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
                image,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                (255, 255, 255),
                cv2.FILLED,
            )

            cv2.putText(
                image,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image)
