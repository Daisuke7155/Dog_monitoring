import streamlit as st
import pandas as pd
import time

# データの読み込み関数
@st.cache
def load_data():
    return pd.read_csv('action_durations.csv')

# データの更新関数
def update_data(interval=60):
    while True:
        data = load_data()
        st.write(f"Data updated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.dataframe(data)
        time.sleep(interval)

# Streamlitアプリのレイアウト
st.title("Dog Monitoring Data")
st.write("This app monitors the dog behavior data periodically.")

# スタートボタン
if st.button('Start Monitoring'):
    interval = st.number_input('Update Interval (seconds)', min_value=10, value=60)
    st.text(f"Data will be updated every {interval} seconds.")
    update_data(interval)
