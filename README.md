# Associative-Classification-Project
빅데이터 분석을 통한 기업신용평가 프로젝트

## The purpose of this project
  + 빅데이터 분석을 통해 기업 신용평가에 기업 리뷰 데이터를 활용
  + 데이터 모델링을 통해 기업 리뷰데이터를 활용을 위한 신뢰성 검사 진행
## Members of this project
  + 1인 프로젝트
  
## The process of this project
  + Jobplanet site에 접근하여 기업 리뷰데이터를 스크래핑
  + 데이터 전처리 작업과 함께 데이터를 정형데이터와 비정형데이터로 분리
  + 정형 데이터는 리뷰에 대한 별점(총점, 복지점수, 문화점수, 다양성 점수 등)
  + 비정형 데이터는 기업의 장, 단점, 경영진에게 바라는 것 등등 텍스트로 이루어져 있다.
  + 정형 데이터에 대한 데이터 분류 진행
  + 비정형 데이터에 대한 NLP 진행
  + 데이터 모델 정확도가 90퍼센트 이상이면 기업의 리뷰를 신용평가에 사용

## 정형데이터 수집 및 분석
  + 데이터 수집
    + chrome에 접근하기 위한 드라이버 다운받기
    + 다운받은 드라이버를 불러오기
      ```
       driver = webdriver.Chrome("/Users/eungchan/desktop/apply/chromedriver")
      ```
    + 드라이버를 이용하여 잡플래닛 사이트에 접속하기
      ```
      driver.get("https://www.jobplanet.co.kr/users/sign_in?_nav=gb")
      ```
    + 미리 가입한 아이디와 비밀번호를 이용하여 로그인 후 리뷰 페이지로 이동
      ```
      login_id = driver.find_element_by_css_selector("input#user_email")
      login_id.send_keys(usr)
      login_pwd = driver.find_element_by_css_selector("input#user_password")
      login_pwd.send_keys(pwd)
      login_id.send_keys(Keys.RETURN)
      
      search_query = driver.find_element_by_css_selector("input#search_bar_search_query")
      search_query.send_keys(query)
      search_query.send_keys(Keys.RETURN)
      ```
    + 데이터 크롤링 및 전처리
  + 전체 데이터 중 90%를 train data를 이용하여 decision tree model 구현
  + 모델은 다음과 같은 것을 예측할 수 있다.
  + 승진기회 AA, 복지 BB, 워라벨 A, 기업문화 AA이면, 최종 등급은 AA
  + 나머지 10% 데이터를 test data로 정하고 최종등급 값을 따로 저장
  + 최종등급이 없는 test data의 리뷰 점수와 새로 만들어진 리뷰점수가 90% 이상 일치하면 기업의 리뷰데이터의 신뢰도를 입증

## 데이터 모델링
  + Decision Tree를 빌드하여 모델을 생성
  + info gain을 이용한 Tree 빌드
    ```
    def get_InfoGain(data, sub_root, target_name):
    # 전체 엔트로피 계산
    total_entropy = get_entropy(data[target_name])

    # 각 subTree 의 엔트로피 계산
    # unique한 attribute의 리스트를 추출
    sub_trees, counts = np.unique(data[sub_root], return_counts=True)
    # 각각의 리스트에 대한 확률 계산
    p = get_prob(counts)
    # 각각의 attribute에 대한 확률에 그 attribute의 엔트로피를 곱한 값들의 합이
    # sub_root의 전체 info-gain이 된다.
    # dataframe.where를 이용해 전체 sub tree에서 각 branch에 해당하는 값들만 추출
    # where 조건에 해당하지 않는 나머지 값들(Nan)은 dropna()를 이용해 제거
    # 남은 데이터 중 target_column이 가지고 있는 데이터만 추출하여 entropy함수에 인자로 전달
    sub_entropy = 0
    for i in range(len(sub_trees)):
        sub_entropy += p[i] * get_entropy(
            data.where(data[sub_root] == sub_trees[i]).dropna()[target_name])

    Info_Gain = total_entropy - sub_entropy
    return Info_Gain
    ```
  + info gain을 계산하기 위해 사용되는 loss function은 entropy
    ```
    def get_entropy(an_attribute):
    # 임이의 attibute을 넘겨 받아, 이 attribute이 포함하는 데이터를 중
    # unique한 값들의 리스트와 그 개수를 리턴
    unique_elements, counts = np.unique(an_attribute, return_counts=True)
    # unique한 값들의 개수를 포함한 리스트를 이용해 확률 계산
    p = get_prob(counts)
    # 각각의 값들에 대한 확률은 리스트로 전달되어
    # 확률 리스트를 인덱싱하면서 entropy 계산
    entropy = 0
    for i in range(len(unique_elements)):
        entropy += -(p[i] * np.log2(p[i]))
    return entropy
    ```
 ## 모델 평가 
    <img width="793" alt="스크린샷 2021-12-27 오후 11 38 48" src="https://user-images.githubusercontent.com/83147205/147481826-b7a078cd-da6c-4dd7-a918-7b0944460565.png">

