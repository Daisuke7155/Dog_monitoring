import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み関数
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/Daisuke7155/dog_monitoring/main/action_durations.csv'
    data = pd.read_csv(url)
    return data

# データの更新関数
def update_data():
    data = load_data()
    st.write(f"Data updated at {pd.Timestamp.now()}")
    st.dataframe(data)
    plot_action_durations(data)

# 行動時間の合計をプロットする関数
def plot_action_durations(data):
    data['Start Time'] = pd.to_datetime(data['Start Time'])
    data['End Time'] = pd.to_datetime(data['End Time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])

    # 行動ごとの合計時間を計算
    action_durations = data.groupby('Action')['Duration (s)'].sum()

    fig, ax = plt.subplots()
    action_durations.plot(kind='bar', ax=ax)
    plt.xlabel('Action')
    plt.ylabel('Total Duration (s)')
    plt.title('Total Duration of Each Action Over the Day')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Streamlitアプリのレイアウト
st.title("Dog Monitoring Data")
st.write("This app monitors the dog behavior data periodically and plots total durations of each action over the day.")

# データの更新ボタン
if st.button('Update Data'):
    update_data()

# 初回のデータ表示
update_data()
