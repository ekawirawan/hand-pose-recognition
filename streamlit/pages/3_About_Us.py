import streamlit as st

st.set_page_config(
    page_title="About Us",
    page_icon=":camera_with_flash:",
)

st.title("About Us")

col1, col2 = st.columns(2)
st.markdown("---")
col3, col4 = st.columns(2)

with col1:
    st.image("../assets/illustrations/selfie_illustration.jpg")
with col2:
    st.markdown("**Hello there!** :wave:")
    st.markdown(
        "I'm :blue[**I Putu Eka Wirawan**], a passionate explorer in the vast realm of technology, currently navigating my academic journey as a student."
    )
    st.markdown(
        "Join me on this thrilling adventure where bytes and bits come to life, and let's explore the boundless horizons of technology together! 🚀✨"
    )
    st.link_button("Github", "https://github.com/ekawirawan")


with col3:
    st.markdown("**Hello everyone!** :raised_hands:")
    st.markdown(
        "I'm :blue[**I Ketut Wisnadiputra**], a passionate explorer in the vast realm of technology, currently navigating my academic journey as a student."
    )
    st.markdown(
        "Join me on this thrilling adventure where bytes and bits come to life, and let's explore the boundless horizons of technology together! 🚀✨"
    )
    st.link_button("Github", "https://github.com/wisnadiputra1")
with col4:
    st.image("../assets/illustrations/selfie_illustration.jpg")

import streamlit as st

st.set_page_config(
    page_title="About Us",
    page_icon=":camera_with_flash:",
)

st.header("About Us", divider="rainbow")

col1, col2 = st.columns(2)
st.markdown("---")
col3, col4 = st.columns(2)

with col1:
    st.image("assets/illustrations/selfie_illustration.jpg")
with col2:
    st.markdown("**Hello there!** :wave:")
    st.markdown(
        "I'm :blue[**I Putu Eka Wirawan**], a passionate explorer in the vast realm of technology, currently navigating my academic journey as a student."
    )
    st.markdown(
        "Join me on this thrilling adventure where bytes and bits come to life, and let's explore the boundless horizons of technology together! 🚀✨"
    )
    st.link_button("Github", "https://github.com/ekawirawan")


with col3:
    st.markdown("**Hello everyone!** :raised_hands:")
    st.markdown(
        "I'm :blue[**I Ketut Wisnadiputra**], a passionate explorer in the vast realm of technology, currently navigating my academic journey as a student."
    )
    st.markdown(
        "Join me on this thrilling adventure where bytes and bits come to life, and let's explore the boundless horizons of technology together! 🚀✨"
    )
    st.link_button("Github", "https://github.com/wisnadiputra1")
with col4:
    st.image("assets/illustrations/selfie_illustration.jpg")