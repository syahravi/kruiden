from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

app = FastAPI()

# Load and preprocess dataset
df = pd.read_csv("DatasetHerbal.csv")
nama_herbal_column_name = 'Nama Herbal'
khasiat_column_name = 'Khasiat'
saran_penyajian_column_name = 'Saran Penyajian'

# Save the original case of the columns
df['original_case_nama_herbal'] = df[nama_herbal_column_name]
df['original_case_khasiat'] = df[khasiat_column_name]
df['original_case_saran'] = df[saran_penyajian_column_name]

# Convert all words to lowercase
df[nama_herbal_column_name] = df[nama_herbal_column_name].str.lower()
df[khasiat_column_name] = df[khasiat_column_name].str.lower()
df[saran_penyajian_column_name] = df[saran_penyajian_column_name].str.lower()

# Combine 'Khasiat' and 'Saran Penyajian' columns into one text
df['combined_text'] = df[nama_herbal_column_name] + ' ' + df[khasiat_column_name] + ' ' + df[saran_penyajian_column_name]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

# Function to recommend rempah based on user input
def recommend_rempah(input_text):
    input_text = input_text.lower()
    input_vector = tfidf_vectorizer.transform([input_text])

    cosine_similarities = linear_kernel(input_vector, tfidf_matrix).flatten()

    # Filter out recommendations with cosine similarity = 0.0
    nonzero_indices = [i for i, sim in enumerate(cosine_similarities) if sim != 0.0]
    if not nonzero_indices:
        return None

    # Sort and get top 7 recommendations
    related_rempah_indices = sorted(nonzero_indices, key=lambda i: cosine_similarities[i], reverse=True)[:7]

    recommended_rempah = df.iloc[related_rempah_indices][['original_case_nama_herbal', 'original_case_khasiat', 'original_case_saran']]
    return recommended_rempah.to_dict(orient='records')  # Return recommended rempah as a list of dictionaries

class InputPayload(BaseModel):
    penyakit: str

@app.post("/recommend", response_model=list)
async def get_recommendation(payload: InputPayload):
    result = recommend_rempah(payload.penyakit)

    if result is not None:
        return result
    else:
        raise HTTPException(status_code=404, detail=f'Tidak ada rekomendasi yang sesuai untuk penyakit {payload.penyakit}.')