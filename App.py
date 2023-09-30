
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from API import transfer_style
import time


# Set page configs. Get emoji names from WebFx
st.set_page_config(page_title="NEURAL STYLE TRANSFER(CONTENT + STYLE= NEW IMAGE)",
                   page_icon="./assets/favicon.png", layout="centered")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 50px;font-weight: 350;font-family:Cursive "> NEURAL STYLE TRANSFER </p>'
st.markdown(title, unsafe_allow_html=True)


st.markdown(
    "<b> <i> Create the IMAGE FROM NST using Machine Learning ! </i> </b>  &nbsp; We takes 2 images — Content Image & Style Image — and blends "
    "OUTPUT WILL BE CONTENT OF FIRST IMAGE AND STYLE OF SECOND IMAGE", unsafe_allow_html=True
)


# Example Image
st.image(image="./assets/nst.png")
st.markdown("</br>", unsafe_allow_html=True)


# -------------Sidebar Section------------------------------------------------


with st.sidebar:

    st.image(image="./assets/speed-brush.gif")
    st.markdown("</br>", unsafe_allow_html=True)

    st.markdown('<p style="font-size: 25px;font-weight: 550;">RESULTS FROM TRAINED MODEL</p>',
                unsafe_allow_html=True)
    st.markdown('Below are some of the art we created using NST.',
                unsafe_allow_html=True)

    # ---------------------Example art images------------------------------

    col1, col2 = st.columns(2)
    with col1:
        st.image(image="./assets/content1.jpg")
    with col2:
        st.image(image="./assets/art1.png")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image="./assets/content2.jpg")
    with col2:
        st.image(image="./assets/art2.png")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image="./assets/content3.jpg")
    with col2:
        st.image(image="./assets/art3.png")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image="./assets/content4.jpg")
    with col2:
        st.image(image="./assets/art4.png")

 # ----------------------------------------------------------------------

    # line break
    st.markdown(" ")
    # About the programmer
    st.markdown("## Made by **OUR TEAM** \U0001F609")


# -------------Body Section------------------------------------------------

# Upload Images
col1, col2 = st.columns(2)
content_image = None
style_image = None
with col1:
    content_image = st.file_uploader(
        "Upload Content Image (PNG & JPG images only)", type=['png', 'jpg'])
with col2:
    style_image = st.file_uploader(
        "Upload Style Image (PNG & JPG images only)", type=['png', 'jpg'])


st.markdown("</br>", unsafe_allow_html=True)
st.warning('NOTE : You need atleast Intel i3 with 8GB memory. ' +
   ' Images greater then (2000x2000) are resized to (1000x1000).')


if content_image is not None and style_image is not None:

        prg=st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            prg.progress(i+1)

        content_image = Image.open(content_image)
        style_image = Image.open(style_image)

        # Convert PIL Image to numpy array
        content_image = np.array(content_image)
        style_image = np.array(style_image)

        # Path of the pre-trained TF model
        model_path = r"model"

        # output image
        styled_image = transfer_style(content_image, style_image, model_path)
        if style_image is not None:
            st.balloons()

        col1, col2 = st.columns(2)
        with col1:
            # Display the output
            st.image(styled_image)
        with col2:

            st.markdown("</br>", unsafe_allow_html=True)
            st.markdown(
                "<b> YOU CAN DOWNLOAD RESULTED (CONTENT IMAGE & STYLE IMAGE). </b>", unsafe_allow_html=True)

            # de-normalize the image
            styled_image = (styled_image * 255).astype(np.uint8)
            # convert to pillow image
            img = Image.fromarray(styled_image)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            st.download_button(
                label="Download",
                data=buffered.getvalue(),
                file_name="NST_IMAGE.png",
                mime="image/png")
