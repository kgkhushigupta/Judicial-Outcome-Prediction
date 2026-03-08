import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# download nltk resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------------
# 1️⃣ Load JSON datasets
# -----------------------------------

files = [
    "constitution_qa.json",
    "crpc_qa.json",
    "ipc_qa.json"
]

data_list = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        data_list.extend(data)

# convert to dataframe
df = pd.DataFrame(data_list)

print("Dataset loaded successfully")
print(df.head())
print("Columns:", df.columns)

# -----------------------------------
# 2️⃣ Clean the Legal Text
# -----------------------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = nltk.word_tokenize(str(text).lower())
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# legal text column
df["Clean_Text"] = df["input"].apply(clean_text)

# -----------------------------------
# 3️⃣ TF-IDF Keyword Extraction
# -----------------------------------

vectorizer = TfidfVectorizer(max_features=10)

X = vectorizer.fit_transform(df["Clean_Text"])

keywords = vectorizer.get_feature_names_out()

print("Extracted Keywords:")
print(keywords)

# -----------------------------------
# 4️⃣ Create Keywords Column
# -----------------------------------

def extract_keywords(text):
    words = text.split()
    return ", ".join(words[:5])

df["Keywords"] = df["Clean_Text"].apply(extract_keywords)

# -----------------------------------
# 5️⃣ Prepare Final Dataset
# -----------------------------------

final_df = pd.DataFrame()

final_df["Case_Text"] = df["input"]
final_df["Keywords"] = df["Keywords"]
final_df["Verdict"] = df["output"]

# -----------------------------------
# 6️⃣ Save Processed Dataset
# -----------------------------------

final_df.to_csv("processed_cases.csv", index=False)

print("Processed dataset saved as processed_cases.csv")
print(final_df.head())