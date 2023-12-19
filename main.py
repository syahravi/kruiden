from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
import secrets
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import List

app = FastAPI()

# Simpan kunci rahasia
SECRET_KEY = "mysecretkey"

# OAuth2PasswordBearer untuk otentikasi token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fungsi untuk menghasilkan token
def create_access_token(data: dict):
    return secrets.token_urlsafe(32)

# Load and preprocess dataset (assuming it's already loaded)
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

# Fungsi untuk merekomendasikan rempah berdasarkan input pengguna
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

    recommended_rempah = df.iloc[related_rempah_indices][['Nama Herbal', 'Khasiat', 'Saran Penyajian']]
    return recommended_rempah.to_dict(orient='records')  # Return recommended rempah as a list of dictionaries

class InputPayload(BaseModel):
    penyakit: str

# Endpoint untuk mendapatkan token
@app.post("/token")
async def login_for_access_token(form_data: InputPayload):
    if form_data.penyakit == "testuser" and form_data.penyakit == "testpassword":
        access_token = create_access_token(data={"sub": form_data.penyakit})
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Endpoint terlindungi yang memerlukan token
@app.post("/recommend", response_model=List[dict])
async def get_recommendation(
    payload: InputPayload, token: str = Depends(oauth2_scheme)
):
    result = recommend_rempah(payload.penyakit)

    if result is not None:
        return result
    else:
        raise HTTPException(
            status_code=404,
            detail=f'Tidak ada rekomendasi yang sesuai untuk penyakit {payload.penyakit}.'
        )
