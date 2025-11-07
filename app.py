import os
import json
import joblib
import pandas as pd
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
import os
os.environ["GEMINI_API_KEY"] = "AIzaSyBEGCigc9cjT0zjKyxed3k_lNJWLvqnfo4"


# =========================================
# FUNGSI CHAT AI (OpenAI & Gemini)
# =========================================
def chat_reply(system_prompt: str, messages: list, api_key: str, model="gpt-4o-mini", provider="openai"):
    """
    Fungsi untuk mengirim percakapan ke OpenAI atau Gemini
    """
    if provider == "openai":
        import openai
        openai.api_key = api_key
        conversation = [{"role": "system", "content": system_prompt}] + messages
        resp = openai.ChatCompletion.create(
            model=model,
            messages=conversation,
            max_tokens=600,
            temperature=0.2,
        )
        return resp.choices[0].message["content"].strip()

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        prompt_text = system_prompt + "\n\n"
        for m in messages:
            role = m["role"]
            content = m["content"]
            prompt_text += f"{role.upper()}: {content}\n"
        model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model_gemini.generate_content(prompt_text)
        return response.text.strip()

    else:
        raise ValueError("Provider tidak dikenali. Gunakan 'openai' atau 'gemini'.")

# =========================================
# FUNGSI MODEL
# =========================================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

# =========================================
# KONFIGURASI STREAMLIT
# =========================================
st.set_page_config(page_title="Prediksi Harga Mobil + Chat AI", layout="wide")
st.title("🚗 Prediksi Harga Mobil & Chat AI")

# Sidebar
st.sidebar.header("⚙️ Konfigurasi")
model_path = st.sidebar.text_input("Path ke model (.pkl)", value="model.pkl")
example_json_path = st.sidebar.text_input("Path contoh fitur (example_features.json)", value="example_features.json")

openai_key_env = st.sidebar.text_input("Masukkan API Key (opsional)", value="", type="password")
ai_provider = st.sidebar.selectbox("Pilih provider AI", ["openai", "gemini"], index=0)

# ✅ Perbaikan bagian ENV KEY
if ai_provider == "gemini":
    use_env_key = st.sidebar.checkbox("Gunakan dari ENV var GEMINI_API_KEY jika kosong", value=True)
else:
    use_env_key = st.sidebar.checkbox("Gunakan dari ENV var OPENAI_API_KEY jika kosong", value=True)

if use_env_key and not openai_key_env:
    if ai_provider == "gemini":
        openai_key_env = os.environ.get("GEMINI_API_KEY", "")
    else:
        openai_key_env = os.environ.get("OPENAI_API_KEY", "")

# =========================================
# LOAD MODEL & SCHEMA
# =========================================
example_schema = {}
try:
    with open(example_json_path, "r") as f:
        example_schema = json.load(f)
except Exception as e:
    st.sidebar.warning(f"Gagal load example_features.json: {e}. Gunakan default.")
    example_schema = {
        "model": "Aygo",
        "year": 2017,
        "transmission": "Manual",
        "mileage": 11730,
        "fuelType": "Petrol",
        "tax": 0,
        "mpg": 68.9,
        "engineSize": 1.0
    }

try:
    model = load_model(model_path)
    st.sidebar.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"Gagal memuat model dari {model_path}: {e}")
    st.stop()

# =========================================
# LAYOUT UTAMA
# =========================================
left, right = st.columns([1.2, 1.0])

# =========================================
# KIRI: INPUT FITUR & PREDIKSI (Pajak Otomatis 🇮🇩)
# =========================================
with left:
    st.subheader("📊 Input Fitur Mobil (Versi Indonesia)")

    # 📦 Estimasi pajak default per model
    pajak_default = {
        "Toyota Avanza": 2000000,
        "Daihatsu Xenia": 1800000,
        "Honda Brio": 1500000,
        "Toyota Agya": 1300000,
        "Mitsubishi Xpander": 2500000,
        "Suzuki Ertiga": 2200000,
        "Toyota Rush": 2600000,
        "Honda Jazz": 2300000,
        "Daihatsu Terios": 2400000,
        "Toyota Yaris": 2300000,
        "Honda HR-V": 2800000,
        "Toyota Fortuner": 3500000,
        "Mitsubishi Pajero Sport": 3800000,
        "Toyota Innova": 3000000,
        "Honda CR-V": 3400000,
        "Suzuki Baleno": 2100000,
        "Toyota Raize": 1900000,
        "Honda BR-V": 2500000,
        "Daihatsu Sigra": 1400000,
        "Wuling Almaz": 2700000
    }

    # 🔁 Simpan pajak di session_state biar bisa auto update
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Toyota Avanza"
    if "current_tax" not in st.session_state:
        st.session_state.current_tax = pajak_default[st.session_state.selected_model]

    with st.form("input_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            model_name = st.selectbox(
                "Model",
                list(pajak_default.keys()),
                index=list(pajak_default.keys()).index(st.session_state.selected_model)
            )

            # update pajak otomatis kalau model berubah
            if model_name != st.session_state.selected_model:
                st.session_state.selected_model = model_name
                st.session_state.current_tax = pajak_default[model_name]

            year = st.number_input("Tahun", min_value=1990, max_value=2025, value=2020)
            transmission = st.selectbox("Transmisi", ["Manual", "Otomatis (Automatic)", "CVT"], index=0)
            fuel_type = st.selectbox("Jenis Bahan Bakar", ["Bensin", "Diesel", "Hybrid", "Listrik"], index=0)

        with col2:
            mileage = st.number_input("Jarak Tempuh (km)", min_value=0, value=20000, step=5000)
            tax = st.number_input("Pajak (Rp)", min_value=0, max_value=10000000, 
                                  value=st.session_state.current_tax, step=500000)
            mpg = st.number_input("Efisiensi BBM (km/liter)", min_value=0.0, value=14.0, step=0.5)
            engine_size = st.number_input("Kapasitas Mesin (Liter)", min_value=0.0, value=1.5, step=0.1)

        submitted = st.form_submit_button("🚗 Prediksi Harga Mobil")

    if submitted:
        input_dict = {
            "model": model_name,
            "year": year,
            "transmission": transmission,
            "mileage": mileage,
            "fuelType": fuel_type,
            "tax": tax,
            "mpg": mpg,
            "engineSize": engine_size
        }

        X_df = pd.DataFrame([input_dict])

        try:
            pred = model.predict(X_df)
            price = float(pred[0]) * 20000  # konversi ke Rupiah
            st.success(f"💰 Prediksi harga mobil: **Rp{price:,.0f}**")
            st.session_state["last_prediction"] = {"price": price, "input": input_dict}
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

        # Tombol Analisis Otomatis
        if openai_key_env and st.button("🤖 Analisis Otomatis dengan AI"):
            system_prompt = (
                f"Kamu adalah asisten AI yang menjelaskan hasil prediksi harga mobil.\n"
                f"Model memprediksi harga £{price:,.2f} untuk mobil dengan fitur: {json.dumps(input_dict, ensure_ascii=False)}."
            )
            try:
                ai_reply = chat_reply(system_prompt, [], openai_key_env, provider=ai_provider)
                st.info(ai_reply)
            except Exception as e:
                st.error(f"Gagal menghubungi AI: {e}")
        elif not openai_key_env:
            st.warning("Masukkan API Key di sidebar atau file .env untuk menggunakan fitur AI otomatis.")

    if st.checkbox("📋 Tampilkan data input"):
        st.dataframe(pd.DataFrame([{
            "model": model_name,
            "year": year,
            "transmission": transmission,
            "mileage": mileage,
            "fuelType": fuel_type,
            "tax": tax,
            "mpg": mpg,
            "engineSize": engine_size
        }]))

# =========================================
# KANAN: CHAT AI (versi estetik & gak kaku 🌸)
# =========================================
with right:
    st.subheader("💬 Chat dengan AI (Analisis & Penjelasan)")

    # CSS untuk bubble chat
    st.markdown("""
        <style>
        .user-bubble {
            background-color: #ffcce7;
            color: #333;
            padding: 12px 18px;
            border-radius: 20px 20px 5px 20px;
            margin: 8px 0;
            text-align: right;
            width: fit-content;
            max-width: 90%;
            float: right;
            clear: both;
            font-size: 15px;
            word-wrap: break-word;
        }
        .ai-bubble {
            background-color: #f1f1f1;
            color: #222;
            padding: 12px 18px;
            border-radius: 20px 20px 20px 5px;
            margin: 8px 0;
            text-align: left;
            width: fit-content;
            max-width: 90%;
            float: left;
            clear: both;
            font-size: 15px;
            word-wrap: break-word;
        }
        </style>
    """, unsafe_allow_html=True)

    # simpan riwayat chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # tampilkan chat kalau ada
    if st.session_state.chat_history:
        st.markdown('<div style="background-color:#fff;border-radius:15px;padding:15px;border:1px solid #ddd;box-shadow:0 2px 8px rgba(0,0,0,0.05);height:400px;overflow-y:auto;">', unsafe_allow_html=True)
        for role, text in st.session_state.chat_history:
            if role == "user":
                st.markdown(f'<div class="user-bubble">{text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-bubble">{text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # input chat
    user_input = st.chat_input("Tulis pesan di sini 💭 (tekan Enter untuk kirim)")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        system_prompt = (
            "Kamu adalah asisten AI yang hangat, responsif, dan santai. "
            "Gunakan bahasa ringan dan ramah seperti ngobrol dengan teman, boleh pakai emoji kecil 😊✨. "
            "Kamu bisa menjawab berbagai hal, tapi kalau tentang mobil, kasih jawaban super detail "
            "(spesifikasi, harga pasaran di Indonesia, keunggulan, dan kekurangannya). "
            "Pastikan jawabannya enak dibaca, jangan kaku, dan tetap sopan."
        )
        
        messages = [{"role": "user" if r == "user" else "assistant", "content": t} for r, t in st.session_state.chat_history[-10:]]

        if not openai_key_env:
            st.error("❌ API key belum diatur.")
        else:
            try:
                ai_reply = chat_reply(system_prompt, messages, openai_key_env, provider=ai_provider)
                st.session_state.chat_history.append(("assistant", ai_reply))
                st.rerun()  # biar langsung refresh kayak chat beneran
            except Exception as e:
                st.error(f"Gagal menghubungi AI: {e}")
