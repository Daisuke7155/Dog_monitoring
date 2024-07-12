import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# データの読み込み関数
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/Daisuke7155/dog_monitoring/main/action_durations.csv'
    data = pd.read_csv(url)
    return data

# データの更新関数
def update_data(interval=60):
    while True:
        data = load_data()
        st.write(f"Data updated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.dataframe(data)
        
        # 行動回数のプロット
        plot_action_counts(data)
        
        time.sleep(interval)

# 行動回数のプロット関数
def plot_action_counts(data):
    data['Start Time'] = pd.to_datetime(data['Start Time'])
    data['End Time'] = pd.to_datetime(data['End Time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])

    action_counts = data.groupby([data['Start Time'].dt.floor('min'), 'Action']).size().unstack(fill_value=0)

    fig, ax = plt.subplots()
    action_counts.plot(ax=ax)
    plt.xlabel('Time')
    plt.ylabel('Action Count')
    plt.title('Action Counts Over Time')
    st.pyplot(fig)

# Streamlitアプリのレイアウト
st.title("Dog Monitoring Data")
st.write("This app monitors the dog behavior data periodically and plots action counts over time.")

# スタートボタン
if st.button('Start Monitoring'):
    interval = st.number_input('Update Interval (seconds)', min_value=10, value=60)
    st.text(f"Data will be updated every {interval} seconds.")
    update_data(interval)
