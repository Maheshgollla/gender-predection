import nltk
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import names
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
)

# Download dataset
nltk.download('names')

# Feature extraction function
def extract_features(name):
    name = name.lower()                                                 
    
    return {
        'last_letter': name[-1],
        'last_two': name[-2:],
        'first_letter': name[0],
        'length': len(name),
        'vowel_count': sum(1 for char in name if char in 'aeiou')
    }

# Load and shuffle dataset
labeled_names = [(name, 'male') for name in names.words('male.txt')] + \
                [(name, 'female') for name in names.words('female.txt')]
random.shuffle(labeled_names)

# Prepare dataset
features = [extract_features(name) for name, gender in labeled_names]
labels = [gender for _, gender in labeled_names]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
model = make_pipeline(DictVectorizer(sparse=False), MultinomialNB())
model.fit(X_train, y_train)

# API request model
class NameRequest(BaseModel):
    name: str

# API endpoint to predict gender
@app.post("/predict")
def predict_gender(request: NameRequest):
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Invalid name")
    prediction = model.predict([extract_features(name)])[0]
    return {"gender": prediction}

# Run the server with: uvicorn main:app --reload
