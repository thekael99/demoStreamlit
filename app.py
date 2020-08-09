from keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image
import numpy as np
import os


def main():
    model = load_model('my_cifar10_model.h5')
    number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    st.title("CIFAR-10 - Object Recognition")
    st.text("Streamlit + Keras")
    st.text(number_to_class)
    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select activity", activities)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if choice == "Detection":
        st.subheader("Object")
        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if image_file is not None:
            img = Image.open(image_file)
            st.text("Your image")
            st.image(img)
            if st.button("Detec!"):
                # st.text("Processing...")
                my_image_resized = img.resize((32, 32))
                data = np.array(my_image_resized).reshape((1, 32, 32, 3))
                probabilities = model.predict(data)
                index = np.argsort(probabilities[0, :])
                st.text("Most likely class: " + str(number_to_class[index[9]]))
    else:
        st.subheader("Kael99 - Anttizen, Demo streamlit.")


if __name__ == "__main__":
    main()
