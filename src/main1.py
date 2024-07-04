# ================================================================================================================================




import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import easyocr
from PIL import Image
from colorthief import ColorThief
import matplotlib.pyplot as plt
import numpy as np
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PIL import Image
import torch
import numpy as np


# ==============================================================================================================================



def get_color_palette(img_path: str) -> list:
    try:
        ct = ColorThief(img_path)
        palette = ct.get_palette(color_count=5)
        plt.imshow([[palette[i] for i in range(5)]])
        plt.show()
        return palette
    except Exception as e:
        print(f"Encountered error {e} in construction of color palette")
        return []
    

# =================================================Analize Palette==============================================================




def analyze_palette(palette: list) -> int:
    score = 0
    if not palette:
        return score

    # Example scoring logic based on color vibrancy and harmony
    for color in palette:
        if np.mean(color) > 128:  # Check if the color is generally bright
            score += 2
        else:
            score += 1

    # Further adjust score based on diversity and harmony
    if len(set(palette)) > 3:  # Diverse palette
        score += 2

    return score




# ==================================================Extract_text===================================================================

def get_overlay_box(image_path: str) -> str:
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    return ' '.join([text for (_, text, _) in result])



# ==================================================Analize_Sentiment==================================================================



def analyze_sentiment(text: str) -> int:

# Interpret the sentiment score

#****VADER (Valence Aware Dictionary and sEntiment Reasoner): VADER is a rule-based sentiment analysis tool specifically tuned 
# for social media sentiment analysis. It's known for handling emoticons, slang, and other nuances in text.****


    analyzer = SentimentIntensityAnalyzer()

    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
     sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
       sentiment = "Negative"
    else:
      sentiment = "Neutral"

# Output results

    return sentiment 



# ================================================Analize interactive object===============================================================



def analyze_objects(objects: list) -> int:
    score = 0
    if not objects:
        return score

    # Example scoring logic based on common ad components
    key_objects = ['person', 'logo', 'product', 'car', 'bicycle','banana']
    for obj in objects:
        if obj in key_objects:
            score += 2

    return score



# =================================================layout Extraction for promotion==========================================================



# Function to load YOLOv5 model
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

# Function to detect objects using YOLOv5
def detect_objects(img_path, model):
    results = model(img_path)
    boxes = results.xyxy[0].cpu().numpy()  # Move to CPU and then convert to numpy

    return boxes  # Return bounding boxes

# Function to calculate layout features
def calculate_layout_features(boxes, image_size):
    features = {
        'alignment': 0,
        'symmetry': 0,
        'spacing': 0,
        'element_size': 0
    }

    if len(boxes) == 0:
        return features

    width, height = image_size
    areas = []
    centroids = []

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        area = (x2 - x1) * (y2 - y1)
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        areas.append(area)
        centroids.append(centroid)

    # Alignment: Check if objects are aligned along vertical or horizontal axis
    vertical_alignment = all(abs(c[0] - centroids[0][0]) < width * 0.1 for c in centroids)
    horizontal_alignment = all(abs(c[1] - centroids[0][1]) < height * 0.1 for c in centroids)
    features['alignment'] = int(vertical_alignment or horizontal_alignment)

    # Symmetry: Check symmetry around the image center
    center_x, center_y = width / 2, height / 2
    symmetry_scores = [
        abs((2 * center_x - c[0]) - c[0]) + abs((2 * center_y - c[1]) - c[1]) for c in centroids
    ]
    features['symmetry'] = int(np.mean(symmetry_scores) < (width + height) * 0.1)

    # Spacing: Check if objects are evenly spaced
    distances = [
        np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) for i, c1 in enumerate(centroids) for c2 in centroids[i + 1:]
    ]
    features['spacing'] = int(np.std(distances) < min(width, height) * 0.1)

    # Element Size: Check if elements have similar sizes
    features['element_size'] = int(np.std(areas) < np.mean(areas) * 0.5)

    return features

# Function to analyze layout
def analyze_layout(img_path):
    model = load_model()
    image = Image.open(img_path)
    image_size = image.size
    boxes = detect_objects(img_path, model)
    layout_features = calculate_layout_features(boxes, image_size)

    # Aggregate layout score
    layout_score = sum(layout_features.values())
    return layout_score




# ====================================================Analize_text score for Promotion==============================================================




def analyze_text(text):
    # Define promotional keywords
    promotional_keywords = ["sale", "discount", "buy now", "offer", "limited time", "save", "special", "exclusive","product","products","delivered","starts"]
    keyword_score = sum(1 for word in text.split() if word.lower() in promotional_keywords)

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    sentiment_score = sentiment_scores['compound']+sentiment_scores['pos']+sentiment_scores['neu']
   
 
    # Combine keyword and sentiment score
    text_score = keyword_score * 2 + sentiment_score 
    return text_score, sentiment_scores




# ======================================================Analize_Object score for promotion==============================================================



def analyze_objects1(objects):
    # Define promotional objects
    promotional_objects = ["product", "logo", "sale sign", "price tag"]
    object_score1 = sum(1 for obj in objects if obj in promotional_objects)
    return object_score1


