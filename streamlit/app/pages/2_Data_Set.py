import os

import streamlit as st

st.set_page_config(
    page_title="Datasets",
    page_icon=":camera_with_flash:",
)


def load_images(path):
    images = []
    for file_name in os.listdir(path):
        if file_name.endswith(".jpg"):
            images.append(os.path.join(path, file_name))

    return images


st.title("Datasets")

path = "../assets/datasets"

images = load_images(path)

categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# Initialize st.session_state.filter_option
if "filter_option" not in st.session_state:
    st.session_state.filter_option = "All"

# Menambahkan observer pada selectbox
filter_option = st.selectbox(
    "Choose Category", ["All"] + categories, key="category_selector"
)

# Set st.session_state.indexSelected menjadi 0 ketika filter_option berubah
if filter_option != st.session_state.filter_option:
    st.session_state.indexSelected = 0
    st.session_state.filter_option = filter_option

if filter_option == "All":
    displayed_images = []
    for category in categories:
        category_folder = os.path.join(path, category)
        displayed_images.extend(load_images(category_folder))
else:
    category_folder = os.path.join(path, filter_option)
    displayed_images = load_images(category_folder)

if "indexSelected" not in st.session_state:
    st.session_state.indexSelected = 0

if not displayed_images:
    st.warning("No images match the selected filter.")
else:
    col1, col2 = st.columns(2)
    with col1:
        prevBtn = st.button("Prev")
    with col2:
        nextBtn = st.button("Next")

    if nextBtn:
        if st.session_state.indexSelected < len(displayed_images) - 1:
            st.session_state.indexSelected += 1

    if prevBtn:
        if st.session_state.indexSelected > 0:
            st.session_state.indexSelected -= 1

    st.write(f"{st.session_state.indexSelected + 1} / {len(displayed_images)}")
    st.image(displayed_images[st.session_state.indexSelected], use_column_width=True)
