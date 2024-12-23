from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime


category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']  # 뉴스 카테고리 리스트


url = 'https://news.naver.com/section/100'  # 기본 URL

df_titles = pd.DataFrame()  # 빈 데이터프레임 생성#

for i in range(6):  # 6개 카테고리에 대해 반복
   url = 'https://news.naver.com/section/10{}'.format(i)  # 각 카테고리별 URL 생성
   resp = requests.get(url)  # URL로 요청 보내기
   soup = BeautifulSoup(resp.text, 'html.parser')  # HTML 파싱
   title_tags = soup.select('.sa_text_strong')  # 뉴스 제목 태그 선택
   titles = []  # 제목 저장할 리스트
   for title_tag in title_tags:  # 각 제목 태그에 대해
       title = title_tag.text  # 텍스트 추출
       title = re.compile('[^가-힣]').sub(' ', title)  # 한글 외 문자를 'p'로 대체
       titles.append(title)  # 리스트에 제목 추가
   df_section_titles = pd.DataFrame(titles, columns=['titles'])  # 제목으로 데이터프레임 생성
   df_section_titles['category'] = category[i]  # 카테고리 열 추가
   df_titles = pd.concat([df_titles, df_section_titles], axis='rows', ignore_index=True)  # 데이터프레임 합치기

print(df_titles.head())  # 상위 5행 출력
df_titles.info()  # 데이터프레임 정보 출력
print(df_titles['category'].value_counts())  # 카테고리별 개수 출력
df_titles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(
   datetime.datetime.now().strftime('%Y%m%d')), index=False)  # CSV 파일로 저장 (파일명에 날짜 포함)
