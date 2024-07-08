# LIKITH_ZOCKET

**Instructions to Run the Application**
Install Dependencies:
Make sure you have Python 3.8 or later installed. Navigate to your project directory and run:

    pip install -r requirements.txt
    
Run the Application:
Use the Streamlit CLI to run the application:

    streamlit run main.py
    
Upload and Analyze Images:
**Open the URL provided by Streamlit (usually http://localhost:8501).
**Upload images for analysis.
**View results, provide feedback, and interact with the database.

**Oject Detection:YOLOv8
**TexExtraction:EasyOCR
**Sentiment Analysis:VADER
**Promotional ad or non promotional ad:Transformer model(interesting part of this project)

Code Organization

main.py:
    Contains the main Streamlit application code.

src:
    Directory containing helper functions and modules for color palette extraction, text detection, object detection, layout analysis, and sentiment analysis.

Database:
     SQLite database ad_analyzer.db to store analysis history and user feedback.

By following these instructions and using the provided requirements.txt file, you can set up and run the Ad Creativity Analyzer application smoothly.


![Screenshot 2024-07-04 161322](https://github.com/likithkumar003/LIKITH_ZOCKET/assets/133403175/73de3192-3056-4c1b-83a1-a24368337cfb)

uploded example image:

![Screenshot 2024-07-04 161701](https://github.com/likithkumar003/LIKITH_ZOCKET/assets/133403175/91c1d327-1eb7-4fdd-96f2-6fe7d84b10c3)

Can see whether the add is creative or not:
![Screenshot 2024-07-04 161731](https://github.com/likithkumar003/LIKITH_ZOCKET/assets/133403175/df73cc58-e267-43cd-b1a4-f754feca2ee2)

Analysis:
![Screenshot 2024-07-04 161829](https://github.com/likithkumar003/LIKITH_ZOCKET/assets/133403175/cbe268a4-b61a-4cb1-bcc6-57ab7d1b9071)

Database:
![Screenshot 2024-07-04 161842](https://github.com/likithkumar003/LIKITH_ZOCKET/assets/133403175/3abb1491-e2d4-44d0-98aa-ec46dbfc7171)
