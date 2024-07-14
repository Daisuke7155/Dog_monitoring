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
    plot_cumulative_action_durations(data)
    st.dataframe(data)

# 尿分析データの更新関数
def update_urine_data():
    urine_data_url = 'https://raw.githubusercontent.com/Daisuke7155/dog_monitoring/main/urine_data.csv'  # 仮のURL
    data = load_data(urine_data_url)
    st.write(f"Urine data updated at {pd.Timestamp.now()}")
    plot_urine_analysis(data)
    st.dataframe(data)

# 行動回数をカウントする関数
def count_actions(data):
    data['Start Time'] = pd.to_datetime(data['Start Time'])
    data['End Time'] = pd.to_datetime(data['End Time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])
    
    actions = ['drinking', 'defecating', 'urinating']
    action_counts = {action: 0 for action in actions}

    for action in actions:
        action_data = data[data['Action'] == action]
        if not action_data.empty:
            action_data = action_data.sort_values('Start Time').reset_index(drop=True)
            action_count = 0
            ongoing = False
            for i in range(len(action_data)):
                if action_data.loc[i, 'Duration (s)'] >= 1:
                    if not ongoing:
                        action_count += 1
                        ongoing = True
                    else:
                        if i > 0 and (action_data.loc[i, 'Start Time'] - action_data.loc[i-1, 'End Time']).total_seconds() > 0.5:
                            action_count += 1
            action_counts[action] = action_count

    st.subheader('Count of Specific Actions')
    for action, count in action_counts.items():
        st.write(f'{action}: {count} times')

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

# 各時刻に対する各行動の積算時間をプロットする関数
def plot_cumulative_action_durations(data):
    data['Start Time'] = pd.to_datetime(data['Start Time'])
    data['End Time'] = pd.to_datetime(data['End Time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])

    # 各行動の積算時間を計算
    cumulative_data = data.copy()
    cumulative_data['Cumulative Duration (s)'] = cumulative_data.groupby('Action')['Duration (s)'].cumsum()

    fig, ax = plt.subplots()
    for action in cumulative_data['Action'].unique():
        action_data = cumulative_data[cumulative_data['Action'] == action]
        action_data = action_data.sort_values('Start Time')
        ax.plot(action_data['Start Time'], action_data['Cumulative Duration (s)'], label=action)

    plt.xlabel('Time')
    plt.ylabel('Cumulative Duration (s)')
    plt.title('Cumulative Duration of Each Action Over Time')
    plt.legend()
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
page = st.sidebar.radio("Select a Page", ["Home", "Behavior Analysis", "Urinary Analysis"])

if page == "Home":
    st.write("Welcome to the Dog Monitoring Data App. Use the sidebar to navigate to different sections.")
elif page == "Behavior Analysis":
    st.write("This section provides an analysis of the dog's behavior data.")
    if st.button('Update Behavior Data'):
        update_behavior_data()
        data = load_data('https://raw.githubusercontent.com/Daisuke7155/dog_monitoring/main/action_durations.csv')
        count_actions(data)
elif page == "Urinary Analysis":
    st.write("This section provides an analysis of the dog's urine data.")
    if st.button('Update Urine Data'):
        update_urine_data()
