import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon=":camera_with_flash:",
)

col1, col2 = st.columns(2)

with col1:
    st.title("Make Your Selfie Experience :blue[Enjoyable]")
    st.markdown(" ")
    st.markdown(
        "Try a camera filter using one of your favorite hand poses :+1: :v: :sign_of_the_horns:"
    )
    st.link_button("Try now :sparkles:", "/Camera_Filter_✨")
with col2:
    st.image("assets/illustrations/selfie_illustration.jpg")
