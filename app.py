import os
import json
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Google Sheetsからデータを読み込む関数
def load_data_from_sheets(sheet_name):
    try:
        # 環境変数から認証情報を取得
        credentials_info = os.getenv('GOOGLE_CREDENTIALS')
        if not credentials_info:
            st.error("GOOGLE_CREDENTIALS is not set.")
            return None

        credentials_info = json.loads(credentials_info)

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
        gc = gspread.authorize(credentials)

        spreadsheet_key = os.getenv('SPREADSHEET_KEY')
        if not spreadsheet_key:
            st.error("SPREADSHEET_KEY is not set.")
            return None

        st.write(f"Loading data from sheet: {sheet_name}")  # Debug message
        worksheet = gc.open_by_key(spreadsheet_key).worksheet(sheet_name)
        data = worksheet.get_all_records()
        if not data:
            st.error(f"No data found in sheet: {sheet_name}")
            return None
        df = pd.DataFrame(data)
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Spreadsheet not found. Please check the spreadsheet key and ensure the service account has access.")
        return None
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        return None

# 行動データの更新関数
def update_behavior_data():
    data = load_data_from_sheets("Behavior Data")
    if data is not None:
        st.write(f"Behavior data updated at {pd.Timestamp.now()}")

        st.markdown("## 📊 Behavior Analysis")
        
        st.markdown("### Count of Specific Actions")
        action_counts = count_actions(data)
        for action, count in action_counts.items():
            st.markdown(f"**{action.capitalize()}**: {count} times")
        
        st.markdown("---")
        
        st.markdown("### Cumulative Duration of Each Action Over Time")
        plot_cumulative_action_durations(data)
        
        st.markdown("---")
        
        st.markdown("### Count of Each Action Over Time")
        plot_action_counts_over_time(data)
        
        st.markdown("---")
        
        st.markdown("### Raw Data")
        st.dataframe(data)

# 尿分析データの更新関数
def update_urine_data():
    data = load_data_from_sheets("Urine Data")
    if data is not None:
        st.write(f"Urine data updated at {pd.Timestamp.now()}")
        plot_urine_analysis(data)
        st.dataframe(data)

# 行動回数をカウントする関数
def count_actions(data):
    data['Start time'] = pd.to_datetime(data['Start time'])
    data['End time'] = pd.to_datetime(data['End time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])
    
    actions = ['drinking', 'defecating', 'urinating', 'awake']
    action_counts = {action: 0 for action in actions}

    for action in actions:
        action_data = data[data['Action'] == action]
        if not action_data.empty:
            action_data = action_data.sort_values('Start time').reset_index(drop=True)
            action_count = 0
            ongoing = False
            for i in range(len(action_data)):
                if action_data.loc[i, 'Duration (s)'] >= 1:
                    if not ongoing:
                        action_count += 1
                        ongoing = True
                    else:
                        if i > 0 and (action_data.loc[i, 'Start time'] - action_data.loc[i-1, 'End time']).total_seconds() > 0.5:
                            action_count += 1
            action_counts[action] = action_count

    return action_counts

# 行動回数を日付ごとにプロットする関数
def plot_action_counts_over_time(data):
    data['Start time'] = pd.to_datetime(data['Start time'])
    data['End time'] = pd.to_datetime(data['End time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])
    data['Date'] = data['Start time'].dt.date
    
    actions = ['drinking', 'defecating', 'urinating', 'awake']
    action_counts = {action: [] for action in actions}
    dates = sorted(data['Date'].unique())
    
    for date in dates:
        daily_data = data[data['Date'] == date]
        daily_counts = count_actions(daily_data)
        for action in actions:
            action_counts[action].append(daily_counts[action])
    
    fig, ax = plt.subplots()
    for action in actions:
        ax.plot(dates, action_counts[action], 'o-', label=action)  # ドットでプロット
    
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Count of Each Action Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 各時刻に対する各行動の積算時間をプロットする関数
def plot_cumulative_action_durations(data):
    data['Start time'] = pd.to_datetime(data['Start time'])
    data['End time'] = pd.to_datetime(data['End time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])

    # 各行動の積算時間を計算
    cumulative_data = data.copy()
    cumulative_data['Cumulative Duration (s)'] = cumulative_data.groupby('Action')['Duration (s)'].cumsum()

    fig, ax = plt.subplots()
    for action in cumulative_data['Action'].unique():
        action_data = cumulative_data[cumulative_data['Action'] == action]
        action_data = action_data.sort_values('Start time')
        ax.plot(action_data['Start time'], action_data['Cumulative Duration (s)'], label=action)

    plt.xlabel('Time')
    plt.ylabel('Cumulative Duration (s)')
    plt.title('Cumulative Duration of Each Action Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 尿分析データをプロットする関数（仮の内容）
def plot_urine_analysis(data):
    st.subheader('Urine Analysis Data')
    st.write("Plotting urine analysis data is not yet implemented.")

# リアルタイム動画を表示する関数
def display_real_time_video():
    st.subheader('Real-Time Video Feed')
    run = st.checkbox('Run')
    if run:
        stframe = st.empty()
        while True:
            # Raspberry PiからMJPEGストリームを取得
            stream_url = "http://<Raspberry_Pi_IP>:8080/?action=stream"  # <Raspberry_Pi_IP>をRaspberry PiのIPアドレスに置き換えてください
            response = requests.get(stream_url, stream=True)
            if response.status_code == 200:
                bytes_data = b''
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        img = Image.open(BytesIO(jpg))
                        stframe.image(img, use_column_width=True)
            else:
                st.write("Failed to get video stream")

# Streamlitアプリのレイアウト
st.title("Dog Monitoring Data")

# サイドバーにページのリンクを追加
page = st.sidebar.radio("Select a Page", ["Home", "Behavior Analysis", "Urinary Analysis", "Real-Time Video"])

if page == "Home":
    st.write("Welcome to the Dog Monitoring Data App. Use the sidebar to navigate to different sections.")
    st.image("home.png", caption="Home Image")
elif page == "Behavior Analysis":
    st.write("Behavior Analysis")
    if st.button('Update Behavior Data'):
        update_behavior_data()
elif page == "Urinary Analysis":
    st.write("This section provides an analysis of the dog's urine data.")
    if st.button('Update Urine Data'):
        update_urine_data()
elif page == "Real-Time Video":
    display_real_time_video()
