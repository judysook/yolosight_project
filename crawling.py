from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.request
import time

# 크롬 드라이버 서비스 설정
service = Service(executable_path="C:/Users/user/Documents/OSS_proj/crawling/chromedriver-win64/chromedriver-win64/chromedriver.exe")
driver = webdriver.Chrome(service=service)

# 사용자로부터 검색어 입력 받기
search_keyword = input("검색어를 입력하세요: ")
directory_keyword = input("디렉토리 명을 입력하세요: ")
# 네이버 이미지 검색 URL 생성 ("검색어"에 따른 결과 페이지로 바로 이동)
url = "https://search.naver.com/search.naver?where=image&sm=tab_jum&query=" + search_keyword
driver.get(url)

# 페이지가 로드되어 https로 시작하는 이미지들이 나타날 때까지 대기
WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.XPATH, "//img[starts-with(@src, 'https')]"))
)

# 눈에 보이지 않는 이미지들도 로드하기 위해 페이지 스크롤 다운 (필요한 경우 딜레이와 횟수 조절)
body = driver.find_element(By.TAG_NAME, "body")
for _ in range(10):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)

# 이미지 크롤링: XPath를 활용하여 src 속성이 https로 시작하는 모든 이미지 요소 선택
# 중복 URL이 없도록 리스트에 추가
images = driver.find_elements(By.XPATH, "//img[starts-with(@src, 'https')]")
img_urls = []
for img in images:
    src = img.get_attribute("src")
    if src and src not in img_urls:
        img_urls.append(src)

print(f"찾은 이미지의 개수: {len(img_urls)}")

# 이미지 다운로드: 원하는 폴더 경로에 이미지 저장 (다운로드 중 에러가 발생할 경우 예외 처리)
for index, img_url in enumerate(img_urls):
    try:
        urllib.request.urlretrieve(img_url, f"C:/Users/user/Documents/OSS_proj/crawling/{directory_keyword}_images/{search_keyword}_image{index}.jpg")
    except Exception as e:
        print(f"이미지 {index} 다운로드 실패: {e}")

print("다운로드 완료.")
driver.quit()
