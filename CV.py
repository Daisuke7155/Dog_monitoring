import time
from gpiozero import MCP3008
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os
import statistics


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
    spreadsheet_key_path = './config/spreadsheet_key.json'

    # 認証情報のJSONファイルを読み込む
    with open(credentials_path) as f:
        credentials_info = json.load(f)

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
    gc = gspread.authorize(credentials)

    # スプレッドシートキーの JSON ファイルを読み込む
    with open(spreadsheet_key_path) as f:
        spreadsheet_key_data = json.load(f)
        spreadsheet_key = spreadsheet_key_data.get('SPREADSHEET_KEY')

    if not spreadsheet_key:
        raise ValueError("SPREADSHEET_KEY is not set in the JSON file.")

    try:
        worksheet = gc.open_by_key(spreadsheet_key).worksheet("pH")  # 正しいシート名に置き換える
        return worksheet
    except gspread.exceptions.SpreadsheetNotFound as e:
        print(f"Spreadsheet not found: {e}")
        raise
    except gspread.exceptions.WorksheetNotFound as e:
        print(f"Worksheet not found: {e}")
        raise

# MCP3008 の CH0 ピンに接続されたアナログセンサーを読み取る
adc = MCP3008(channel=0)
ec = DFRobot_EC()

def main():
    worksheet = init_google_sheets()
    temperature_data = []
    ec_data = []
    data = []

    print("Press 'f' to finish measuring and upload data.")

    try:
        while True:
            voltage = adc.value * 5 * 1000  # ミリボルトに変換
            temperature = ec.temperature  # 固定温度値
            ecValue = ec.readEC(voltage, temperature)

            print(f"Temperature: {temperature:.1f} °C")
            print(f"EC: {ecValue:.2f} mS/m")

            temperature_data.append(temperature)
            ec_data.append(ecValue)
            data.append([time.strftime('%Y-%m-%d %H:%M:%S'), temperature, ecValue])

            ec.calibration(voltage, temperature)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Measurement interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if temperature_data and ec_data:
            avg_temperature = statistics.mean(temperature_data)
            avg_ecValue = statistics.mean(ec_data)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            worksheet.append_row([timestamp, avg_temperature, avg_ecValue])
            print("Average data uploaded to Google Sheets.")
        if data:
            worksheet.append_rows(data)
            print("Data uploaded to Google Sheets.")
        else:
            print("No data to upload.")

if __name__ == "__main__":
    main()
