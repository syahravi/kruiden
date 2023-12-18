import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

output = 'Dataset Herbal.csv'

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

# Menggabungkan kolom 'Khasiat' dan 'Saran Penyajian' menjadi satu teks
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

    # Sort and get top 10 recommendations
    related_rempah_indices = sorted(nonzero_indices, key=lambda i: cosine_similarities[i], reverse=True)[:7]

    recommended_rempah = df.iloc[related_rempah_indices][['original_case_nama_herbal', 'original_case_khasiat', 'original_case_saran']]
    return recommended_rempah # Kembalikan rekomendasi rempah

# User input through Colab GUI
input_penyakit = input("Masukkan penyakit: ")
result = recommend_rempah(input_penyakit)

if result is not None:
    recommendations = result
    print("\n")
    # Outputkan hasil per baris
    for index, row in recommendations.iterrows():
        print(f"Nama Herbal: {row['original_case_nama_herbal']}")
        print(f"Khasiat: {row['original_case_khasiat']}")
        print(f"Saran Penyajian: {row['original_case_saran']}")

        print("\n" + "="*500 + "\n")  # Pembatas antar rekomendasi
else:
    print(f'Maaf, tidak ada rekomendasi yang sesuai untuk penyakit {input_penyakit}.')