# ============================================================================================

import streamlit as st
import torch
import clip
from PIL import Image

# =============================================================================================

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ==============================================================================================

# Define your labels
labels = ["promotional ad", "non-promotional image"]

def classify_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return labels[probs.argmax()]

# ===============================================================================================

# Streamlit Interface
def main():
    st.title("Ad Promotional Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img_path = "temp_image.jpg"
        image.save(img_path)

        if st.button('Classify'):
            with st.spinner("Classifying..."):
                result = classify_image(img_path)
            st.write(f"The image is classified as: **{result}**")

# ================================================================================================

if __name__ == "__main__":
    main()
