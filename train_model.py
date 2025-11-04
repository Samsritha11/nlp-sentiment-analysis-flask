# ===============================
# 🧠 Train and Save NLP Model
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
import pickle

# Download stopwords (only first time)
nltk.download('stopwords')

# ---------- Dataset ----------
data = {
    'text': [
        "I absolutely loved this movie — the story was emotional and beautifully told.",
        "What a waste of two hours. The script was lazy and predictable.",
        "Brilliant acting and direction! Every scene felt alive and purposeful.",
        "The film dragged on forever, I nearly fell asleep halfway through.",
        "An enjoyable family drama with a heartfelt message and great music.",
        "Terrible editing and weak dialogues ruined what could have been a great film.",
        "The visuals were stunning, and the cinematography was top-notch.",
        "I didn’t hate it, but it definitely wasn’t worth the hype.",
        "An inspiring story that left me smiling for hours afterward.",
        "The movie felt all over the place with no clear direction or focus.",
        "Incredible performances — especially by the lead actress.",
        "Too slow and unoriginal. I’ve seen better movies on TV.",
        "A touching story that made me cry at the end.",
        "Completely overhyped. The trailer was better than the movie itself.",
        "A solid romantic comedy with genuine laughs and likable characters.",
        "Mediocre plot but the visuals and soundtrack saved it.",
        "Not bad, but it didn’t leave much of an impression either.",
        "One of the most powerful films I’ve watched this year.",
        "Disappointing finale — it felt rushed and unfinished.",
        "A simple yet deeply moving story about friendship and loss.",
        "It was fine, nothing extraordinary but not terrible either.",
        "Absolutely terrible acting and no chemistry between the leads.",
        "The humor was forced and cringeworthy throughout.",
        "A heartwarming movie that’s perfect for a Sunday evening.",
        "The special effects were impressive, but the plot lacked soul.",
        "A complete mess. I couldn’t wait for it to end.",
        "I’d definitely watch this one again — pure joy from start to finish!",
        "Good performances, but the pacing was inconsistent.",
        "Predictable storyline, yet somehow still entertaining.",
        "This film deserves an award — beautifully made in every aspect.",
        "Couldn’t connect with any of the characters; they felt flat.",
        "A masterpiece in storytelling and emotional depth.",
        "Not my kind of movie, but I respect the effort behind it.",
        "Poor direction made the talented cast look clueless.",
        "An absolute joy — I was smiling the entire time!",
        "It had potential, but weak writing let it down.",
        "Decent film overall, though not something I’d recommend to everyone.",
        "A breathtaking performance by the supporting actor.",
        "Just average, nothing memorable at all.",
        "A brilliant balance of humor, heart, and humanity."
    ],
    'label': [
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'neutral', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'positive',
        'neutral', 'neutral', 'positive', 'negative', 'positive',
        'neutral', 'negative', 'negative', 'positive', 'neutral',
        'negative', 'positive', 'neutral', 'positive', 'positive',
        'negative', 'positive', 'neutral', 'negative', 'positive',
        'neutral', 'neutral', 'positive', 'neutral', 'positive'
    ]
}

df = pd.DataFrame(data)

# ---------- Clean Text ----------
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['text'].str.lower().apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# ---------- Vectorize ----------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# ---------- Train Model ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------- Save Model ----------
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved successfully!")
