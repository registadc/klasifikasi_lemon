import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_lemon.joblib")

st.set_page_config(
	page_title = "Klasifikasi Lemon",
	page_icon = ":tangerine:"
)

st.title("Klasifikasi Lemon")
st.markdown("Aplikasi machine learning untuk klasifikasi lemon")

diameter = st.slider("diameter",50,100,70)
berat = st.slider("berat",70,150,100)
tebal_kulit = st.slider("tebal_kulit",1.0,10.0,5.0)
kadar_gula = st.slider("kadar_gula",4.0,10.0,5.0)
asal_daerah = st.pills("asal_daerah", ["California","Malang","Medan"], default="California")
musim_panen = st.pills("musim_panen", ["Puncak","Akhir","Awal"], default="Puncak")
warna = st.pills("warna", ["Hijau pekat","Kuning kehijauan","Kuning cerah"], default="Hijau pekat")

if st.button("prediksi", type="primary"):
	data_baru = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,musim_panen,warna]],
                        columns = ["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","musim_panen","warna"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi{prediksi} dengan tingkat keyakinan {presentase*100:.2f}%")
	st.balloons()

st.divider()
st.caption("dibuat dengan :tangerine: oleh *Regista*")