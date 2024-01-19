import cv2
import streamlit as st


PATH_TO_EMOTE_POSE_V = "assets/illustrations/hand-pose/glasses-pose-v.png"
PATH_TO_EMOTE_THUMB = "assets/illustrations/hand-pose/glasses-thumb.png"
PATH_TO_EMOTE_METAL = "assets/illustrations/hand-pose/glasses-metal.png"


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

    hat_width = int(face_width * 1.0)
    hat_height = int(0.70 * face_height) + 1

    glass_x = x - int((hat_width - face_width) / 2)
    glass_y = y - int(0.10 * face_height)

    glass_resized = cv2.resize(emote, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            if glass_resized[i, j, 3] != 0: 
                fc[glass_y + i][glass_x + j][:3] = glass_resized[
                    i, j, :3
                ]

    return fc


def handEmote(pose, frame, face_cas, posev_emote, thumb_emote, metal_emote):
    # face_cas = read_face_haarcascade()
    # posev_emote, thumb_emote, metal_emote = read_emote_pose()

    # frame = cv2.imread(img_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_det = face_cas.detectMultiScale(gray, 1.3, 5)

    emote = ""
    for x, y, w, h in face_det:
        if pose == "poseV":
            emote = posev_emote
        elif pose == "thumb":
            emote = thumb_emote
        elif pose == "metal":
            emote = metal_emote
        else:
            print("error")

        frame = put_emote(emote, frame, x, y, w, h)

    return frame
