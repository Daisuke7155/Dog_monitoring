import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み関数
@st.cache
def load_data(url):
    data = pd.read_csv(url)
    return data

# 行動データの更新関数
def update_behavior_data():
    behavior_data_url = 'https://raw.githubusercontent.com/Daisuke7155/dog_monitoring/main/action_durations.csv'
    data = load_data(behavior_data_url)
    st.write(f"Behavior data updated at {pd.Timestamp.now()}")
    plot_action_durations(data)
    st.dataframe(data)

# 尿分析データの更新関数
def update_urine_data():
    urine_data_url = 'https://raw.githubusercontent.com/Daisuke7155/dog_monitoring/main/urine_data.csv'  # 仮のURL
    data = load_data(urine_data_url)
    st.write(f"Urine data updated at {pd.Timestamp.now()}")
    plot_urine_analysis(data)
    st.dataframe(data)

# 行動時間の合計をプロットする関数
def plot_action_durations(data):
    data['Start Time'] = pd.to_datetime(data['Start Time'])
    data['End Time'] = pd.to_datetime(data['End Time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])

    # 行動ごとの合計時間を計算
    action_durations = data.groupby('Action')['Duration (s)'].sum()

    # 各行動の合計時間を表示
    st.subheader('Total Duration of Each Action')
    for action, duration in action_durations.items():
        st.write(f'{action}: {duration} seconds')

    # 棒グラフのプロット
    fig, ax = plt.subplots()
    action_durations.plot(kind='bar', ax=ax)
    plt.xlabel('Action')
    plt.ylabel('Total Duration (s)')
    plt.title('Total Duration of Each Action Over the Day')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 尿分析データをプロットする関数（仮の内容）
def plot_urine_analysis(data):
    # データの仮の処理とプロット
    st.subheader('Urine Analysis Data')
    st.write("Plotting urine analysis data is not yet implemented.")

# Streamlitアプリのレイアウト
st.title("Dog Monitoring Data")

# サイドバーにページのリンクを追加
page = st.sidebar.radio("Select a Page", ["Home", "行動分析", "尿分析"])

if page == "Home":
    st.write("Welcome to the Dog Monitoring Data App. Use the sidebar to navigate to different sections.")
elif page == "Behavior Analysis":
    st.write("This section provides an analysis of the dog's behavior data.")
    if st.button('Update Behavior Data'):
        update_behavior_data()
elif page == "Urinary Analysis":
    st.write("This section provides an analysis of the dog's urine data.")
    if st.button('Update Urine Data'):
        update_urine_data()
