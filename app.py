import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import threading

# データの生成関数（モックデータ）
def generate_data():
    data = {
        "Action": ["drinking", "sleeping", "barking", "defecating"],
        "Start Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 4,
        "End Time": [(datetime.now() + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")] * 4,
        "Duration (s)": [60, 60, 60, 60]
    }
    df = pd.DataFrame(data)
    csv_path = 'data/action_durations.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

# 定期的にデータを生成して表示する関数
def periodic_generate_and_display(interval=60):
    while True:
        csv_path = generate_data()
        df = pd.read_csv(csv_path)
        st.write(f"Data generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.dataframe(df)
        time.sleep(interval)

# Streamlitアプリのレイアウト
st.title("Dog Monitoring Data Generator")
st.write("This app generates and displays dog monitoring data periodically.")

# スタートボタン
if st.button('Start Generating Data'):
    interval = st.number_input('Generation Interval (seconds)', min_value=10, value=60)
    threading.Thread(target=periodic_generate_and_display, args=(interval,), daemon=True).start()
    st.success('Data generation started!')
