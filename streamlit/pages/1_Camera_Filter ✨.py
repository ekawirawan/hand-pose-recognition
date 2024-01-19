import streamlit as st

st.set_page_config(
    page_title="Camera Filter",
    page_icon=":camera_with_flash:",
)

import sys
from helpers.object_detection import realtime_video_detection


sys.path.append("helpers")


st.header("Camera Filter :sparkles:", divider="rainbow")

realtime_video_detection()
