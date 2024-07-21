import time
from gpiozero import MCP3008
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os

class DFRobot_EC:
    def __init__(self):
        self.temperature = 25.0  # 固定温度値

    def readEC(self, voltage, temperature):
        ecValue = voltage  # 例として簡単な変換式
        return ecValue

    def calibration(self, voltage, temperature):
        # キャリブレーションのロジックを追加
        pass

def init_google_sheets():
    # Google 認証情報とスプレッドシートキーのファイルパス
    credentials_path = './config/dogmonitoring-92d60377d8b3.json'
    spreadsheet_key = 'your_spreadsheet_key_here'
    
    # 認証情報のJSONファイルを読み込む
    with open(credentials_path) as f:
        credentials_info = json.load(f)

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
    gc = gspread.authorize(credentials)

    worksheet = gc.open_by_key(spreadsheet_key).worksheet("pH")
    return worksheet

# MCP3008 の CH0 ピンに接続されたアナログセンサーを読み取る
adc = MCP3008(channel=0)
ec = DFRobot_EC()

def main():
    worksheet = init_google_sheets()
    data = []

    print("Press 'f' to finish measuring and upload data.")

    try:
        while True:
            voltage = adc.value * 5 * 1000  # ミリボルトに変換
            temperature = ec.temperature  # 固定温度値
            ecValue = ec.readEC(voltage, temperature)

            print(f"Temperature: {temperature:.1f} °C")
            print(f"EC: {ecValue:.2f} mS/m")

            data.append([time.strftime('%Y-%m-%d %H:%M:%S'), temperature, ecValue])

            ec.calibration(voltage, temperature)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Measurement interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if data:
            worksheet.append_rows(data)
            print("Data uploaded to Google Sheets.")
        else:
            print("No data to upload.")

if __name__ == "__main__":
    main()
