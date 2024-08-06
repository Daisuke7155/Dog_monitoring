import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

def fetch_image_urls(query, max_links_to_fetch, headers):
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # ヘッドレスモードをオンにする場合
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    driver.get(search_url)
    print(f"Opened {search_url}")

    image_urls = set()
    image_count = 0
    scroll_count = 0

    while image_count < max_links_to_fetch:
        # Scroll down to load more images
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Wait for the page to load

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        image_elements = soup.find_all('img')

        for img in image_elements:
            img_url = img.get('src')
            if img_url and img_url not in image_urls:
                image_urls.add(img_url)
                image_count += 1
                if image_count >= max_links_to_fetch:
                    break

        print(f"Found {len(image_urls)} image URLs so far")

        # Check if there are more images to load
        load_more_button = driver.find_elements(By.CSS_SELECTOR, ".mye4qd")
        if load_more_button:
            driver.execute_script("document.querySelector('.mye4qd').click()")
            time.sleep(3)  # Wait for the load more button to fetch images
        else:
            scroll_count += 1
            if scroll_count > 5:  # 5回スクロールしても「もっと見る」ボタンが出ない場合、終了
                print("No more images to load")
                break

    driver.quit()
    return list(image_urls)

def download_images(image_urls, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created folder: {dest_folder}")

    for idx, img_url in enumerate(image_urls):
        for attempt in range(3):  # 最大3回の再試行
            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    timestamp = int(time.time() * 1000)  # タイムスタンプを取得
                    image_path = os.path.join(dest_folder, f"image_{timestamp}_{idx + 1}.jpg")
                    image.save(image_path)
                    print(f"Downloaded {img_url} to {image_path}")
                    time.sleep(0.1)  # ファイル名の重複を防ぐための少しの遅延
                    break  # 成功したらループを抜ける
                else:
                    print(f"Failed to download image: {img_url}, status code: {response.status_code}")
            except Exception as e:
                print(f"Could not download {img_url} - {e}")
                if attempt == 2:
                    print(f"Failed to download image {img_url} after 3 attempts")

def main():
    search_terms = {
        "Shiba Inu drinking": ["dog drink water", "dog drink", "犬 水 飲む", "犬 飲水", "犬 飲む 水"],
        "Shiba Inu sleeping": ["dog sleeping", "dog nap", "犬 睡眠", "犬 寝る", "犬 昼寝"],
        "Shiba Inu barking": ["dog barking", "dog barking at night", "犬 吠える", "犬 吠え", "犬 ムキ"],
        "Shiba Inu defecating": ["dog defecating", "dog pooping", "犬 うんち", "犬 うんこ", "犬 便"],
        "Shiba Inu urinating": ["dog urinating", "dog peeing", "犬 おしっこ", "犬 小便", "犬 しっこ"],
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    # 保存先ディレクトリの指定
    base_folder = input("Enter the base directory to save images: ").strip()

    for folder, queries in search_terms.items():
        dest_folder = os.path.join(base_folder, folder.replace(" ", "_"))
        for query in queries:
            print(f"Fetching images for: {query}")
            image_urls = fetch_image_urls(query, max_links_to_fetch=1000, headers=headers)
            if image_urls:
                download_images(image_urls, dest_folder)
            else:
                print(f"No images found for query: {query}")

if __name__ == "__main__":
    main()
