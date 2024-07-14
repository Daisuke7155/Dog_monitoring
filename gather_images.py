import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time

def fetch_image_urls(query, max_links_to_fetch, headers):
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_elements = soup.find_all('img')

    image_urls = []
    for img in image_elements:
        img_url = img.get('src')
        if img_url and len(image_urls) < max_links_to_fetch:
            image_urls.append(img_url)
    return image_urls

def download_images(image_urls, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for idx, img_url in enumerate(image_urls):
        try:
            response = requests.get(img_url)
            image = Image.open(BytesIO(response.content))
            timestamp = int(time.time() * 1000)  # タイムスタンプを取得
            image_path = os.path.join(dest_folder, f"image_{timestamp}_{idx + 1}.jpg")
            image.save(image_path)
            print(f"Downloaded {img_url} to {image_path}")
            time.sleep(0.1)  # ファイル名の重複を防ぐための少しの遅延
        except Exception as e:
            print(f"Could not download {img_url} - {e}")

def main():
    search_terms = {
        "Shiba Inu drinking": ["Shiba Inu drinking water", "Shiba Inu drinking", "柴犬 水を飲む", "柴犬 飲水", "柴犬 飲み水"],
        "Shiba Inu sleeping": ["Shiba Inu sleeping", "Shiba Inu nap"],
        "Shiba Inu barking": ["Shiba Inu barking", "Shiba Inu barking at night", "柴犬 吠える", "柴犬 夜吠え", "柴犬 夜鳴き"],
        "Shiba Inu defecating": ["Shiba Inu defecating", "Shiba Inu pooping", "柴犬 うんち", "柴犬 うんこ", "柴犬 便"],
        "Shiba Inu urinating": ["Shiba Inu urinating", "Shiba Inu peeing", "柴犬 おしっこ", "柴犬 小便", "柴犬 しっこ"],
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    # 保存先ディレクトリの指定
    base_folder = input("Enter the base directory to save images: ").strip()

    for folder, queries in search_terms.items():
        dest_folder = os.path.join(base_folder, folder.replace(" ", "_"))
        for query in queries:
            print(f"Fetching images for: {query}")
            image_urls = fetch_image_urls(query, max_links_to_fetch=20, headers=headers)
            download_images(image_urls, dest_folder)

if __name__ == "__main__":
    main()
