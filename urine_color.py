import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import json
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Googleサービスアカウントの認証情報を設定
def init_google_sheets():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    with open('./config/dogmonitoring-92d60377d8b3.json') as f:
        credentials_info = json.load(f)
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
    gc = gspread.authorize(credentials)
    
    with open('./config/spreadsheet_key.json') as f:
        spreadsheet_key_info = json.load(f)
    spreadsheet_key = spreadsheet_key_info['SPREADSHEET_KEY']
    
    worksheet = gc.open_by_key(spreadsheet_key).worksheet('urine_color')
    return worksheet

def upload_to_google_sheets(capture_time, most_similar_image, test_hist_path, similar_hist_path):
    worksheet = init_google_sheets()
    worksheet.append_row([capture_time, most_similar_image, test_hist_path, similar_hist_path])

def calculate_histogram(image_path):
    image = cv2.imread(image_path)
    hist = []
    for i in range(3):
        hist.append(cv2.calcHist([image], [i], None, [256], [0, 256]))
    return hist

def compare_histograms(hist1, hist2):
    similarities = []
    for i in range(3):
        similarities.append(cv2.compareHist(hist1[i], hist2[i], cv2.HISTCMP_CORREL))
    return np.mean(similarities)

def plot_histogram(hist, title, save_path):
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        plt.plot(hist[i], color=color)
        plt.xlim([0, 256])
    plt.savefig(save_path)
    plt.close()

def find_most_similar_image(test_image_path, folder_path):
    test_hist = calculate_histogram(test_image_path)
    
    max_similarity = -1
    most_similar_image = None
    most_similar_hist = None

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        comparison_hist = calculate_histogram(image_path)
        similarity = compare_histograms(test_hist, comparison_hist)
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image = image_name
            most_similar_hist = comparison_hist

    return most_similar_image, most_similar_hist

def capture_image_on_keypress():
    cap = cv2.VideoCapture(0)
    print("Press 'f' to capture an image.")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Camera', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            test_image_path = "captured_image.jpg"
            cv2.imwrite(test_image_path, frame)
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return test_image_path

if __name__ == "__main__":
    test_image_path = capture_image_on_keypress()
    urine_folder_path = "./urine"
    most_similar_image, most_similar_hist = find_most_similar_image(test_image_path, urine_folder_path)
    
    if most_similar_image:
        print(f"The most similar image is: {most_similar_image}")
        test_hist = calculate_histogram(test_image_path)
        test_hist_path = "test_image_histogram.png"
        plot_histogram(test_hist, "Test Image RGB Histogram", test_hist_path)
        
        similar_hist_path = "most_similar_image_histogram.png"
        plot_histogram(most_similar_hist, f"Most Similar Image RGB Histogram: {most_similar_image}", similar_hist_path)
        
        capture_time = time.strftime('%Y-%m-%d %H:%M:%S')
        upload_to_google_sheets(capture_time, most_similar_image, test_hist_path, similar_hist_path)
    else:
        print("No similar images found.")
