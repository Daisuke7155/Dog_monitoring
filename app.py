import os
import json
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
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
        
        selected_date = st.selectbox("Select a Date", data['Start time'].str[:10].unique())
        filtered_data = data[data['Start time'].str.contains(selected_date)]

        st.markdown("### Count of Specific Actions")
        action_counts = count_actions(filtered_data)
        for action, count in action_counts.items():
            st.markdown(f"**{action.capitalize()}**: {count} times")
        
        st.markdown("---")
        
        st.markdown("### Cumulative Duration of Each Action Over Time")
        plot_cumulative_action_durations(filtered_data)
        
        st.markdown("---")
        
        st.markdown("### Count of Each Action Over Time")
        plot_action_counts_over_time(data)
        
        st.markdown("---")
        
        st.markdown("### Raw Data")
        st.dataframe(data)

# 尿分析データの更新関数
def update_urine_data():
    data = load_data_from_sheets("CV")
    ph_data = load_data_from_sheets("pH")
    color_data = load_data_from_sheets("urine_color")
    if data is not None:
        st.write(f"Urine data updated at {pd.Timestamp.now()}")
        plot_urine_analysis(data)
        st.dataframe(data)
    if ph_data is not None:
        st.write(f"pH data updated at {pd.Timestamp.now()}")
        plot_ph_analysis(ph_data)
        st.dataframe(ph_data)
    if color_data is not None:
        st.write(f"Urine color data updated at {pd.Timestamp.now()}")
        plot_urine_color_analysis(color_data)
        st.dataframe(color_data)

# 行動回数をカウントする関数
def count_actions(data):
    data['Start time'] = pd.to_datetime(data['Start time'])
    data['End time'] = pd.to_datetime(data['End time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])
    
    actions = ['drinking', 'defecating', 'urinating']
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
    
    actions = ['drinking', 'defecating', 'urinating']
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

# 行動時間の合計をプロットする関数
def plot_action_durations(data):
    data['Start time'] = pd.to_datetime(data['Start time'])
    data['End time'] = pd.to_datetime(data['End time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])

    # 開始時間が同じ時間のDuration (s)を積算
    summed_data = data.groupby(['Start time', 'Action'])['Duration (s)'].sum().reset_index()

    fig, ax = plt.subplots()
    for action in summed_data['Action'].unique():
        action_data = summed_data[summed_data['Action'] == action]
        ax.plot(action_data['Start time'], action_data['Duration (s)'], 'o-', label=action)
    
    plt.xlabel('Time')
    plt.ylabel('Total Duration (s)')
    plt.title('Total Duration of Each Action Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 各時刻に対する各行動の積算時間をプロットする関数
def plot_cumulative_action_durations(data):
    data['Start time'] = pd.to_datetime(data['Start time'])
    data['End time'] = pd.to_datetime(data['End time'])
    data['Duration (s)'] = pd.to_numeric(data['Duration (s)'])

    # 各行動の積算時間を計算
    cumulative_data = data.groupby(['Start time', 'Action'])['Duration (s)'].sum().reset_index()
    cumulative_data['Cumulative Duration (s)'] = cumulative_data.groupby('Action')['Duration (s)'].cumsum()

    fig, ax = plt.subplots()
    for action in cumulative_data['Action'].unique():
        action_data = cumulative_data[cumulative_data['Action'] == action]
        ax.plot(action_data['Start time'], action_data['Cumulative Duration (s)'], label=action)

    plt.xlabel('Time')
    plt.ylabel('Cumulative Duration (s)')
    plt.title('Cumulative Duration of Each Action Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 尿分析データをプロットする関数
def plot_urine_analysis(data):
    data['time'] = pd.to_datetime(data['time'])
    data['cv [mS/m]'] = pd.to_numeric(data['cv [mS/m]'])

    fig, ax = plt.subplots()
    ax.plot(data['time'], data['cv [mS/m]'], 'o-')
    
    # 正常範囲の閾値を追加
    ax.axhline(y=10, color='g', linestyle='--', label='Lower Threshold (10 mS/m)')
    ax.axhline(y=340, color='r', linestyle='--', label='Upper Threshold (340 mS/m)')
    
    # 日付と時間のフォーマット
    date_form = DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(date_form)
    
    plt.xlabel('Time')
    plt.ylabel('Conductivity (mS/m)')
    plt.title('Urine Conductivity Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# pH分析データをプロットする関数
def plot_ph_analysis(data):
    data['time'] = pd.to_datetime(data['time'])
    data['pH'] = pd.to_numeric(data['pH'])

    fig, ax = plt.subplots()
    ax.plot(data['time'], data['pH'], 'o-')
    
    # 正常範囲の閾値を追加
    ax.axhline(y=6.0, color='g', linestyle='--', label='Lower Threshold (6.0)')
    ax.axhline(y=7.0, color='r', linestyle='--', label='Upper Threshold (7.0)')
    
    # 日付と時間のフォーマット
    date_form = DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(date_form)
    
    plt.xlabel('Time')
    plt.ylabel('pH')
    plt.title('Urine pH Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 尿色分析データをプロットする関数
def plot_urine_color_analysis(data):
    data['date'] = pd.to_datetime(data['date'])
    data['color'] = data['color'].str.replace('.jpg', '')

    # カテゴリの順序を指定（OK_normalを最後に表示）
    categories = ['NG_clear', 'NG_green', 'NG_red', 'NG_strong_red', 'OK_normal']
    data['color'] = pd.Categorical(data['color'], categories=categories, ordered=True)
    data['color_code'] = data['color'].cat.codes

    fig, ax = plt.subplots()

    # 境界線を引く
    ok_normal_index = categories.index('OK_normal')
    ax.axhline(y=ok_normal_index - 0.5, color='black', linestyle='--', linewidth=1)

    # 背景色を設定
    ax.axhspan(ymin=ok_normal_index - 0.5, ymax=len(categories), color='lightblue', alpha=0.3)
    ax.axhspan(ymin=0, ymax=ok_normal_index - 0.5, color='lightcoral', alpha=0.3)

    # グラフをプロット
    ax.plot(data['date'], data['color_code'], 'o-')

    # 日付と時間のフォーマット
    date_form = DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(date_form)

    # 縦軸の範囲とラベルを固定
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    plt.xlabel('Date')
    plt.ylabel('Color')
    plt.title('Urine Color Over Time')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 画像表示
    image_folder = './urine'
    st.markdown("### Urine Color Images")
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        if os.path.isfile(image_path):
            st.image(image_path, caption=image_file, use_column_width=True)

def display_real_time_video():
    st.subheader('Real-Time Video Feed')
    run = st.checkbox('Run')
    if run:
        stframe = st.empty()  # 動画フレームの表示スペースを確保
        # グローバルIPアドレスとポートを設定します
        stream_url = "http://126.36.71.194:8080/?action=stream"
        try:
            while run:
                response = requests.get(stream_url, stream=True, timeout=30)
                if response.status_code == 200:
                    bytes_data = b''  # ストリームのバイトデータを保持
                    for chunk in response.iter_content(chunk_size=1024):
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            # 画像データを抽出して表示
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            img = Image.open(BytesIO(jpg))
                            stframe.image(img, use_column_width=True)
                            break  # 次のフレームに進むためループを抜ける
                else:
                    # ステータスコードが200でない場合、接続失敗
                    st.error("No connection")
                    break
        except requests.exceptions.RequestException:
            # 例外が発生した場合、「No connection」と表示
            st.error("No connection")
            
            # ストリーミングが失敗した場合、ローカルの動画ファイルを表示
            st.write("Demo video:")
            if os.path.exists("./image/realtime/demo.mp4"):
                st.video("./image/realtime/demo.mp4")
            else:
                # ローカル動画ファイルが見つからない場合のエラーメッセージ
                st.error("Local video file not found.")
            
# Streamlitアプリのレイアウト
st.title("Dog Monitoring Data")

# サイドバーにページのリンクを追加
page = st.sidebar.radio("Select a Page", ["Home", "Behavior Analysis", "Urinary Analysis", "Real-Time Video"])

if page == "Home":
    st.write("Welcome to the Dog Monitoring Data App. Use the sidebar to navigate to different sections.")
    st.image("./image/home/home.png", caption="Home Image")
    st.image("./image/home/1.png", caption="Issue")
    st.image("./image/home/2.png", caption="Image")
    st.image("./image/home/3.png", caption="Architecture")
    st.image("./image/home/4.png", caption="Business")
    st.image("./image/home/5.png", caption="Future")

elif page == "Behavior Analysis":
    st.write("Behavior Analysis")
    st.image("./image/action/1.png", caption="Behavior analysis")
    update_behavior_data()
elif page == "Urinary Analysis":
    st.write("This section provides an analysis of the dog's urine data.")
    st.image("./image/urine/1.png", caption="Conductivity analysis")
    st.image("./image/urine/2.png", caption="pH analysis")
    st.image("./image/urine/3.png", caption="Color analysis")
    update_urine_data()
elif page == "Real-Time Video":
    st.image("./image/realtime/1.png", caption="Technical details") 
    display_real_time_video()
