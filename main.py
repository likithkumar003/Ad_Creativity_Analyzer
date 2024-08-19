# ==========================================================imports===========================================================================================================


import streamlit as st
import torch
from src.main1 import get_color_palette, get_overlay_box, analyze_palette, analyze_objects, analyze_layout, analyze_sentiment , analyze_objects1,analyze_text
from src.object_det import run_det
from PIL import Image
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from torchvision import transforms, models
from torch import nn
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
from streamlit_lottie import st_lottie
import json



# ================================Lottie animations================================


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

loading_animation = load_lottiefile("animations/loading.json")
creative_animation = load_lottiefile("animations/creative.json")
not_creative_animation = load_lottiefile("animations/not_creative.json")


# ===============================suggesting_4_improvements=========================================


def suggest_improvements(overall_score,palette_score):
    if overall_score <= 15:
        st.warning("This ad is not very creative. Consider the following suggestions:")
        if palette_score <9:
            st.write("- Use brighter and more contrasting colors.")
        st.write("- Add a clear and engaging call-to-action.")
        st.write("- Include attractive visuals or graphics.")

# ==========================================sqlite3======================================================

conn = sqlite3.connect('ad_analyzer.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    rating INTEGER,
    comments TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    palette TEXT,
    text TEXT,
    objects TEXT,
    overall_score REAL,
    sentiment TEXT,
    sentiment_scores TEXT,
    palette_score REAL,
    is_promotional TEXT
)
''')


conn.commit()

# ===================================Initialize session state========================================


if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'user_score' not in st.session_state:
    st.session_state['user_score'] = 0



def display_color_palette(palette):
    plt.figure(figsize=(8, 2))
    plt.imshow([palette])
    plt.axis('off')
    st.pyplot(plt)

# ================================================================================================================

def display_database():
    conn = sqlite3.connect('ad_analyzer.db')
    c = conn.cursor()

    st.subheader("History Table")
    c.execute('SELECT * FROM history')
    history_data = c.fetchall()
    history_df = pd.DataFrame(history_data, columns=['ID', 'Image Path', 'Palette', 'Text', 'Objects', 'Overall Score', 'Sentiment', 'Sentiment Scores', 'Palette Score', 'Is Promotional'])
    gb = GridOptionsBuilder.from_dataframe(history_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    grid_options = gb.build()
    AgGrid(history_df, gridOptions=grid_options)

    st.subheader("Feedback Table")
    c.execute('SELECT * FROM feedback')
    feedback_data = c.fetchall()
    feedback_df = pd.DataFrame(feedback_data, columns=['ID', 'User Name', 'Email', 'Rating', 'Comments', 'Timestamp'])
    gb = GridOptionsBuilder.from_dataframe(feedback_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    grid_options = gb.build()
    AgGrid(feedback_df, gridOptions=grid_options)

    conn.close()



# ===================================================save_to_history===============================================================




def save_to_history(image_path, palette, text, objects, overall_score, sentiment, sentiment_scores, palette_score, is_promotional):
    history_entry = {
        "image": image_path,
        "palette": palette,
        "text": text,
        "objects": objects,
        "scores": overall_score,
        "sentiment1": sentiment,
        "Sentiment_score": sentiment_scores,
        "palette_score": palette_score,
        "is_promotional": is_promotional

    }
    st.session_state['history'].append(history_entry)
    st.session_state['user_score'] += 10
    c.execute('''
    INSERT INTO history (image_path, palette, text, objects, overall_score, sentiment, sentiment_scores, palette_score, is_promotional)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (image_path, str(palette), text, str(objects), overall_score, sentiment, str(sentiment_scores), palette_score, is_promotional))
    conn.commit()




# ===============================generate_detailed_reporte=====================================




def generate_detailed_report(entry):
    st.write(f"### Detailed Report")
    st.image(entry["image"], caption='Analyzed Image', use_column_width=True)
    st.write(f"**Color Palette:**")
    display_color_palette(entry["palette"])
    st.write(f"**Color_score:** {entry['palette_score']}")
    st.write(f"**Overlayed Text:** {entry['text'].strip()}")
    st.write(f"**Detected Objects:** {entry['objects']}")
    st.write(f"**Sentiment Scores:** {entry['Sentiment_score']}")
    st.write(f"**Sentiment:** {entry['sentiment1']}")
    st.write(f"**Is Promotional_by_strategies:** {entry['is_promotional']}")
    st.write("---")





# ====================================determine_if_promotional==================================



def determine_if_promotional(palette_score, text_score, sentiment_scores, object_score1, layout_score):
    overall_score1 = palette_score + text_score + object_score1 + layout_score
    sentiment1 = sentiment_scores['compound']+sentiment_scores['pos']+sentiment_scores['neu']

    if overall_score1 > 10 and sentiment1 > 0.1:
        return "Yes, this is a promotional ad."
    else:
        return "No, this is not a promotional ad."
    


# =============================================main()============================================




def main():
    st.set_page_config(page_title="Ad Creativity Analyzer", page_icon=":art:", layout="wide")

    st.title("Ad Creativity Analyzer")

    st.sidebar.title("User Profile")
    st.sidebar.write(f"Total Score: {st.session_state['user_score']} points")
    if st.session_state['user_score'] >= 50:
        st.sidebar.success("Badge: Creative Analyst")



# ====================================================================================================
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  
    model.load_state_dict(torch.load('promotional_classification_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



# ========================================================================================================

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.subheader("Uploaded Images")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)

    promotional_method = st.sidebar.selectbox(
        "Select promotional_method",
        ("Promotional by Strategies", "Promotional by Transformer Model")
    )        

    if st.button('Analyze'):
        if not uploaded_files:
            st.warning("Please upload at least one image to analyze.")
        else:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                img_path = "temp_image.jpg"
                image.save(img_path)

                # =============================color palette===================================

                with st.spinner("Analyzing...") :
                    st_lottie(loading_animation, height=100, width=100, key="loading")

                    
                    palette = get_color_palette(img_path)
                    palette_score = analyze_palette(palette)

                # ====================Extract text using EasyOCR & sentiment Analysis========================

                    text = get_overlay_box(img_path)
                    sentiment = analyze_sentiment(text)
                    analyzer = SentimentIntensityAnalyzer()
                    sentiment_scores = analyzer.polarity_scores(text)

                # =============================Analize text for Promotional or not==================================  

                    text = get_overlay_box(img_path)
                    text_score, sentiment_scores = analyze_text(text)

                # ==============================Object detection=================================


                    objects = run_det(img_path, 0)
                    object_score = analyze_objects(objects)
                    object_score1=analyze_objects1(objects)

                # ==============================Layout analysis==================================    


                    layout_score = analyze_layout(img_path)

            
                # ================================================================

                    if promotional_method == "Promotional by Transformer Model":
                        input_image = transform(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(input_image)
                            _, preds = torch.max(outputs, 1)
                            is_promotional = "Yes, this is a promotional ad." if preds.item() == 1 else "No, this is not a promotional ad."

                    else:
                        is_promotional = determine_if_promotional(palette_score, text_score, sentiment_scores, object_score1, layout_score)


                    
                    overall_score = (palette_score + sentiment_scores["compound"] + object_score + layout_score) 

            

                # ================================================================    

                    if overall_score >= 11 and text and objects:
                        st.success("This is an Ad creative.")
                        st_lottie(creative_animation, height=200, key="creative")

                    else:
                        st.warning("This is not an Ad creative.")
                        st_lottie(not_creative_animation, height=200, key="not_creative")
                        suggest_improvements(overall_score,palette_score)



                # ================================================================

                    # st.write(f"Model Prediction: {is_promotional_pred}")

                    save_to_history(img_path, palette, text, objects, overall_score, sentiment, sentiment_scores, palette_score, is_promotional)

                # ================================================================
                    
                
            # st.success("Analysis complete!")    

    # ==================================================================================================      

    # Display Analysis History
    st.subheader("Analysis History")
    if st.session_state['history']:
        for idx, entry in enumerate(st.session_state['history']):
            st.write(f"### Analysis {idx + 1}")
            st.image(entry["image"], caption=f'Analysis {idx + 1} Image', use_column_width=True)
            st.write(f"**Color Palette:**")
            display_color_palette(entry["palette"])
            st.write(f"**Overlayed Text**: {entry['text'].strip()}")
            st.write(f"**Detected Objects**: {entry['objects']}")
            # st.write(f"**Scores**: {entry['scores']}")
            st.write(f"**Sentiment**: {entry['sentiment1']}")
            st.write(f"**Is Promotional:** {entry['is_promotional']}")
            # st.write(f"**Is Promotional_Model:** {entry['is_promotional_pred']}")
            if st.button(f"Generate Detailed Report for Analysis {idx + 1}", key=f"report_button_{idx}"):
                generate_detailed_report(entry)


        # ==================================================================================================================


            st.subheader("Feedback")
            name = st.text_input("Name",key="name")
            email = st.text_input("Email")
            rating = st.slider("Rate the accuracy of the analysis", 1, 5, key="rating_slider")
            feedback = st.text_area("Additional comments or suggestions", key="feedback_text_area")

            if st.button('Submit Feedback',key="submit_feedback_button"):
                if name and email:
                    st.success("Thank you for your feedback!")
                    c.execute('''
                    INSERT INTO feedback (name, email, rating, comments)
                    VALUES (?, ?, ?, ?)
                    ''', (name, email, rating, feedback))
                    conn.commit()
                else:
                    st.warning("Please provide your name and email.")        

            
    else:
        st.write("No analysis history available.")


        # ====================================================================================================

    if st.sidebar.button('View Database'):
        display_database()   

    # =====================================================================================================




# =================================================__main__=================================================


if __name__ == "__main__":
    main()
