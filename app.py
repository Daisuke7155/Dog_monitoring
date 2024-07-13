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
    plot_action_counts(data)

# 行動回数のプロット関数
def plot_action_counts(data):
    data['Start Time'] = pd.to_datetime(data['Start Time'])
    data['End Time'] = pd.to_datetime(data['End Time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])

    # 時間ごとに行動の回数をカウント
    action_counts = data.groupby([data['Start Time'].dt.floor('min'), 'Action']).size().unstack(fill_value=0)

    fig, ax = plt.subplots()
    action_counts.plot(kind='bar', ax=ax)
    plt.xlabel('Time')
    plt.ylabel('Action Count')
    plt.title('Action Counts Over Time')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Streamlitアプリのレイアウト
st.title("Dog Monitoring Data")
st.write("This app monitors the dog behavior data periodically and plots action counts over time.")

# データの更新ボタン
if st.button('Update Data'):
    update_data()

# 初回のデータ表示
update_data()
