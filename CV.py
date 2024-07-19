import os
import time
import json
import gspread
from gpiozero import MCP3008
from oauth2client.service_account import ServiceAccountCredentials
import threading

class DFRobot_EC:
    def __init__(self):
        self.temperature = 25.0  # 固定温度値

    def readEC(self, voltage, temperature):
        ecValue = voltage  # 例として簡単な変換式
        return ecValue

    def calibration(self, voltage, temperature):
        # キャリブレーションのロジックを追加
        pass

# Google Sheetsに接続する関数
def init_google_sheets():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials_info = json.loads(os.getenv('GOOGLE_CREDENTIALS'))
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
    gc = gspread.authorize(credentials)
    spreadsheet_key = os.getenv('SPREADSHEET_KEY')
    worksheet = gc.open_by_key(spreadsheet_key).worksheet('pH')
    return worksheet

# 平均値をGoogle Sheetsにアップロードする関数
def upload_to_google_sheets(avg_ec, avg_temp):
    worksheet = init_google_sheets()
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    worksheet.append_row([now, avg_ec, avg_temp])

# MCP3008 の CH0 ピンに接続されたアナログセンサーを読み取る
adc = MCP3008(channel=1)
ec = DFRobot_EC()

def main():
    readings = []
    temperature_readings = []
    print("Press 'f' to finish measuring and upload data.")
    while True:
        voltage = adc.value * 5 * 1000  # ミリボルトに変換
        temperature = ec.temperature  # 固定温度値
        ecValue = ec.readEC(voltage, temperature)

        readings.append(ecValue)
        temperature_readings.append(temperature)

        print(f"Temperature: {temperature:.1f} °C")
        print(f"EC: {ecValue:.2f} mS/m")

        ec.calibration(voltage, temperature)

        time.sleep(1)

        # 測定終了条件
        if input().strip().lower() == 'f':
            break

    avg_ec = sum(readings) / len(readings)
    avg_temp = sum(temperature_readings) / len(temperature_readings)

    print(f"Average Temperature: {avg_temp:.1f} °C")
    print(f"Average EC: {avg_ec:.2f} mS/m")

    # Google Sheetsにアップロード
    upload_to_google_sheets(avg_ec, avg_temp)
    print("Data uploaded to Google Sheets.")

if __name__ == "__main__":
    main()
