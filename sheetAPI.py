from oauth2client.service_account import ServiceAccountCredentials
import gspread
import json

def new_post(name, comment):
    # スコープの作成
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    # 認証情報設定
    # 先ほどのjsonファイルを読み込む
    credentials = ServiceAccountCredentials.from_json_keyfile_name('./config/dogmonitoring-92d60377d8b3.json', scope)
    # サービスアカウントを使ってログインする
    gc = gspread.authorize(credentials)

    # スプレッドシートキーを./config/spreadsheet_key.jsonから読み出す
    with open('./config/spreadsheet_key.json', 'r') as key_file:
        key_data = json.load(key_file)
        SPREADSHEET_KEY = key_data['SPREADSHEET_KEY']

    # シート1で作業行う
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    worksheet.update_cell(1, 1, name) # (1, 1)の値をアップデートする
    worksheet.update_cell(1, 2, comment) # (1, 2)の値をアップデートする

new_post("yama", "2021-07-01")
