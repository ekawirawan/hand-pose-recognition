import cv2
import streamlit as st


PATH_TO_EMOTE_POSE_V = "assets/illustrations\hand-pose\pose-v.png"
PATH_TO_EMOTE_THUMB = "assets/illustrations/hand-pose/pose-thumb.png"
PATH_TO_EMOTE_METAL = "assets\illustrations/hand-pose/pose-metal.png"


@st.cache_resource
def read_emote_pose():
    pose_v = cv2.imread(PATH_TO_EMOTE_POSE_V, cv2.IMREAD_UNCHANGED)
    thumb = cv2.imread(PATH_TO_EMOTE_THUMB, cv2.IMREAD_UNCHANGED)
    metal = cv2.imread(PATH_TO_EMOTE_METAL, cv2.IMREAD_UNCHANGED)

    return pose_v, thumb, metal


@st.cache_resource
def read_face_haarcascade():
    return cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def put_emote(emote, fc, x, y, w, h):
    face_width = w
    face_height = h

    hat_width = face_width - 50
    hat_height = int(0.30 * face_height) + 1

    emote = cv2.resize(emote, (hat_width, hat_height), interpolation=cv2.INTER_NEAREST)

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if emote[i][j][3] > 0:
                    fc[y + i - int(-0.20 * face_height)][x + j][k] = emote[i][j][k]
    return fc


def handEmote(pose, img_path):
    face_cas = read_face_haarcascade()
    posev_emote, thumb_emote, metal_emote = read_emote_pose()

    frame = cv2.imread(img_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_det = face_cas.detectMultiScale(gray, 1.3, 5)

    emote = ""
    for x, y, w, h in face_det:
        if pose == "posev":
            emote = posev_emote
        elif pose == "thumb":
            emote = thumb_emote
        elif pose == "metal":
            emote = metal_emote
        else:
            print("error")

        frame = put_emote(emote, frame, x, y, w, h)

    return frame
