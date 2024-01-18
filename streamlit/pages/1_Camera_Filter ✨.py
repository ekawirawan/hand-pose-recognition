import streamlit as st
import sys

sys.path.append("helpers")
from helpers.object_detection import realtime_video_detection

st.set_page_config(
    page_title="Camera Filter",
    page_icon=":camera_with_flash:",
)

st.header("Camera Filter :sparkles:", divider="rainbow")

realtime_video_detection()
