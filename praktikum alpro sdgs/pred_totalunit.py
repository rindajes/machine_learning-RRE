import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Memuat model dan encoder
model = joblib.load('MLfastfoodunit.pkl')
encoder = joblib.load('encoder.pkl')

st.title("Prediksi Total Unit Tahun (2021)")

# Membuat encoder untuk Fast_Food_Chains jika belum ada
fast_food_chains_encoder = LabelEncoder()
fast_food_chains_encoder.classes_ = np.array(['Arbys', 'Baskin-Robbins', 'Bojangles'])

# Fungsi untuk mendapatkan input dari pengguna
Fast_Food_Chains = st.selectbox('Fast Food Chains :', options=['Arbys', 'Baskin-Robbins', 'Bojangles'])
us_sales = st.number_input("Masukkan U.S. Systemwide Sales (dalam juta dolar): ")
avg_sales = st.number_input("Masukkan Average Sales per Unit (dalam ribuan dolar): ")
franchised_stores = st.number_input("Masukkan jumlah Franchised Stores: ")
company_stores = st.number_input("Masukkan jumlah Company Stores: ")
TotalUnits = st.number_input("Masukkan jumlah TotalUnits: ")
TotalChangeUnits = st.number_input("Masukkan jumlah TotalChange: ", min_value=0, value=100)

if st.button("Prediksi"):
    # Encode Fast_Food_Chains
    fast_food_chains_encoded = fast_food_chains_encoder.transform([Fast_Food_Chains])[0]
    
    # Membuat array data untuk prediksi
    data = np.array([[fast_food_chains_encoded, us_sales, avg_sales, franchised_stores, company_stores, TotalUnits, TotalChangeUnits]])
    try:
        pred_label = model.predict(data)[0]
        pred_unit = encoder.inverse_transform([pred_label])[0]
        # Tampilkan hasil prediksi
        st.success(f"Prediksi Total Units (2021) untuk data baru: {pred_unit}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
