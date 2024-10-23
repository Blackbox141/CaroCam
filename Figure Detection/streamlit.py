import streamlit as st
from PIL import Image

st.title("Chess Game Analyzer")

uploaded_file = st.file_uploader("Upload a chessboard image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Your existing code to process the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    # Display analysis results
    st.write("Best move: e4")
