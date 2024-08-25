import pandas as pd
import os, json, ast
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


class UserState:
    def __init__(self):
        # 초기 상태 설정
        self.state = {
            'model': ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
            'question': None,
            'keywords': {
                'days': None,
                'transport': None,
                'companion': None,
                'theme': None,
                'food': None
            },
            'foods_context': [],
            'playing_context': [],
            'hotel_context': [],
            'scheduler': "",
            'explain': "",
            'second_sentence': "",
            'user_age': "",
            'user_token': "",
            'is_valid': 0,
            #'api_key': os.getenv("OPENAI_API_KEY")
        }

    def get_state(self):
        return self.state


def get_db(kind):
    # 환경 변수에서 데이터베이스 정보 가져오기
    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT", 3306))
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

    if kind == "food":
        sql = """
                SELECT placename, latitude, longitude, teenager, twenties, thirties, fourties, fifties, sixties, 
                       foodtype, bar, parking, operatinghours0, operatinghours1, operatinghours2, operatinghours3, operatinghours4, operatinghours5
                FROM PLACE P JOIN RESTAURANT R ON P.placeid = R.placeid
              """
        # SQL 쿼리를 실행하고 결과를 데이터프레임으로 로드
        df = pd.read_sql(sql, engine)
        return df
    elif kind == "hotel":
        sql = """
                SELECT placename, latitude, longitude, teenager, twenties, thirties, fourties, fifties, sixties, 
                       parking, hoteltype
                FROM PLACE P JOIN HOTEL R ON P.placeid = R.placeid
              """
        # SQL 쿼리를 실행하고 결과를 데이터프레임으로 로드
        df = pd.read_sql(sql, engine)
        return df
    elif kind == "place":
        sql = """
                SELECT placename, latitude, longitude, teenager, twenties, thirties, fourties, fifties, sixties, 
                       single, couple, parents, family, friends, experience, park, uniquetravel, nature, 
                       culture, festival, shopping, history, walking, city, etc, ocean, mountain
                FROM PLACE P JOIN TOUR R ON P.placeid = R.placeid
              """
        df = pd.read_sql(sql, engine)
        return df
    return df


# 사용자 DB를 조회해서 연령대를 결정하는 함수
def get_age_group(age):
    if age < 20:
        return 'teenager'
    elif age < 30:
        return 'twenties'
    elif age < 40:
        return 'thirties'
    elif age < 50:
        return 'fourties'
    elif age < 60:
        return 'fifties'
    else:
        return 'sixties'


def find_keywords(state):
    # 질문과 키워드 기초 상태 받아오기
    question = state['question']
    keywords = state['keywords']
    state['model'] = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    # GPT에게 현재 입력된 정보로부터 키워드 추출 요청
    prompt = ChatPromptTemplate.from_messages([
       ("system", """
        You are tasked with extracting the most relevant information from the user's sentence related to a trip, based on the following categories:

        1. **How many days they want to travel (days)**:
        - Extract the number of days as an integer (e.g., 3).
        
        2. **What transportation they have (transport)**:
        - Identify whether they have a car ('자차') or will use public transportation ('대중교통') such as taxis, buses, or airplanes.

        3. **Who they are traveling with (companion)**:
        - Identify the companion from options like '가족' (family), '부모' (parents), '친구' (friends), '연인' (partner), or '혼자' (alone).

        4. **What theme they prefer for the trip (theme)**:
        - Extract the theme from options like '자연' (nature), '걷기' (walking), '쇼핑' (shopping), '공원' (parks), '이색여행' (unique trips), '문화' (culture), '체험' (experience), '역사' (history), '산' (mountains), '바다' (sea), or '도시' (city).

        5. **What type of food they prefer (food)**:
        - Identify the type of food from options like '한식' (Korean food), '양식' (Western food), '중식' (Chinese food), '일식' (Japanese food), or '아시아' (Asian food).

        **Response**:
        - If some categories are missing from the user's input, ask follow-up questions to gather more information naturally, as if engaging in a conversation. 
        - If all categories are provided, return 'End' at response.

        **Examples**:
        1. If the input is "쉬러 가고 싶어" (I want to rest), suggest themes like '휴식' (rest), and recommend '바다' (sea), '자연' (nature), or '공원' (parks).
        2. If the input is "라면먹고 싶어" (I want to eat ramen), suggest '라면' (ramen) and recommend '한식' (Korean food) or '일식' (Japanese food).
        3. If the input is "사람들을 만나러 가고싶어" (I want to meet people), suggest '도시' (city), '체험' (experience), or '쇼핑' (shopping).
        4. If the input is "아무거나" or "상관없어" or "아무데나" (anything, anywhere), recommend randomly from the provided options.
        5. If the input is "나머지는 추천해줘" or "나머지는 알아서 해줘" (recommend the rest), suggest options for the categories that were not provided.

        Return the information in the following format:
        {
            'days': 'Number of days' or None,
            'transport': '자차/대중교통' or None,
            'companion': '가족/부모/친구/연인/혼자' or None,
            'theme': '자연/걷기/쇼핑/공원/이색여행/문화/체험/역사/산/바다/도시' or None,
            'food': '한식/양식/중식/일식/아시아' or None,
            'response': 'response'
        }
        """),
        ("human","{question}, {keywords}")

    ])

    chain = prompt | state['model']
    response = chain.invoke({"question": question, "keywords": keywords})
    extracted_keywords = ast.literal_eval(response.content)

    # 이미 대답된 정보를 저장
    for key, value in extracted_keywords.items():
        if key in keywords and value is not None and key != 'response':
            if keywords[key] is None:
                keywords[key] = value

    return keywords, extracted_keywords['response']


def foods_search(state):
    user_age = int(state['user_age'])
    data = get_db('food')

    if state['foods_context'] is None:
        state['foods_context'] = []  # None이 아니라 빈 리스트로 초기화

    keywords = state['keywords']
    k = int(state['keywords']['days'])

    def recommend_restaurants(data, condition, user_age, k):
        # 연령대 설정
        if condition['companion'] == '부모':
            user_age_group = 'sixties'
        else:
            user_age_group = get_age_group(user_age)

        # Companion 및 술집 필터링
        if condition['companion'] in ['가족', '부모']:
            data = data[data['bar'] == 0]

        # Time-based Filtering
        def filter_by_time(time_slot):
            if time_slot == '아침':
                return data[
                    (data['operatinghours0'] == 1) | (data['operatinghours1'] == 1) | (data['operatinghours2'] == 1)]
            elif time_slot == '점심':
                return data[(data['operatinghours2'] == 1) | (data['operatinghours3'] == 1)]
            elif time_slot == '저녁':
                return data[(data['operatinghours4'] == 1) | (data['operatinghours5'] == 1)]

        # 가중치 계산
        def calculate_weights(row):
            # Food 가중치
            food_weight = 0.43 if row['foodtype'] == condition['food'] else 0.28

            # 연령대 가중치
            age_score = row[user_age_group]

            # 최종 점수 계산
            return food_weight * age_score

        # 아침, 점심, 저녁 별로 추천 4k개만큼
        recommendations = {}
        recommended_restaurants = set()
        for time_slot in ['아침', '점심', '저녁']:
            filtered_data = filter_by_time(time_slot).copy()  # 명시적으로 복사본을 생성
            filtered_data['FinalScore'] = filtered_data.apply(calculate_weights, axis=1)
            top_restaurants = filtered_data[~filtered_data['placename'].isin(recommended_restaurants)] \
                .sort_values(by='FinalScore', ascending=False).head(k * 4)

            recommendations[time_slot] = list(
                zip(top_restaurants['placename'], top_restaurants['latitude'], top_restaurants['longitude']))

            # 이미 추천된 식당을 추적하여 다음 시간대에 제외
            recommended_restaurants.update(top_restaurants['placename'])

        return recommendations

    response = recommend_restaurants(data, keywords, user_age, k)

    response = state['foods_context'].append(response)

    return state


def hotels_search(state):
    user_age = int(state['user_age'])
    data = get_db('hotel')

    if state['hotel_context'] is None:
        state['hotel_context'] = []  # None이 아니라 빈 리스트로 초기화

    keywords = state['keywords']
    k = int(state['keywords']['days'])

    def recommend_accommodation_with_keywords(data, condition, user_age, k):
        # 연령대 설정
        if condition['companion'] == '부모':
            user_age_group = 'sixties'
        else:
            user_age_group = get_age_group(user_age)

        # 숙박 유형 가중치 설정
        accommodation_weight = {
            '가족': {'호텔': 0.3, '모텔': 0.1, '펜션': 0.6, '게스트하우스': 0},
            '부모': {'호텔': 0.48, '모텔': 0.125, '펜션': 0.395, '게스트하우스': 0},
            '친구': {'호텔': 0.265, '모텔': 0.205, '펜션': 0.26, '게스트하우스': 0.27},
            '연인': {'호텔': 0.27, '모텔': 0.37, '펜션': 0.27, '게스트하우스': 0.09},
            '혼자': {'호텔': 0.263, '모텔': 0.264, '펜션': 0, '게스트하우스': 0.473},
        }

        # 필터링
        filtered_data = data.copy()

        # 교통수단에 따른 주차 필터링
        if condition['transport'] == '자차':
            filtered_data = filtered_data[filtered_data['parking'] == 'Y']

        # 숙박 유형 점수 계산
        filtered_data['typescore'] = filtered_data['hoteltype'].apply(
            lambda x: accommodation_weight[condition['companion']].get(x, 0))

        # 연령대 점수 계산
        filtered_data['age_like'] = filtered_data[user_age_group]

        # 총점 계산
        filtered_data['FinalScore'] = filtered_data['typescore'] * filtered_data['age_like']

        # 총점 기준으로 상위 3k개 숙박지 추천
        recommended = filtered_data.sort_values(by='FinalScore', ascending=False).head(k * 3)

        # 이름, 위도, 경도 리스트로 반환
        result = list(zip(recommended['placename'], recommended['latitude'], recommended['longitude']))
        return result

    response = recommend_accommodation_with_keywords(data, keywords, user_age, k)

    response = state['hotel_context'].append(response)

    return state


def places_search(state):
    user_age = int(state['user_age'])
    data = get_db('place')

    if state['playing_context'] is None:
        state['playing_context'] = []  # None이 아니라 빈 리스트로 초기화

    keywords = state['keywords']
    k = int(state['keywords']['days'])

    def recommend_travel_destinations(data, condition, user_age, k):
        # 연령대 설정
        if condition['companion'] == '부모':
            user_age_group = 'sixties'
        else:
            user_age_group = get_age_group(user_age)

        # Companion 가중치 계산
        if condition['companion'] == '가족':
            comp = 'family'
        elif condition['companion'] == '부모':
            comp = 'parents'
        elif condition['companion'] == '친구':
            comp = 'friends'
        elif condition['companion'] == '연인':
            comp = 'couple'
        elif condition['companion'] == '혼자':
            comp = 'single'
        data['CompanionWeight'] = data[comp].apply(lambda x: 0.6 if x == 1 else 0.1)

        # Theme 그룹 설정
        theme_groups = {
            'nature': ['mountain', 'ocean', 'park', 'walking', 'etc'],
            'city': ['shopping', 'festival', 'culture', 'etc'],
            'history': ['uniquetravel', 'walking', 'culture', 'etc'],
            'uniquetravel': ['experience', 'walking', 'culture', 'park', 'etc'],
            'ocean': ['ocean', 'nature', 'etc'],
            'mountain': ['walking', 'nature', 'etc'],
            'culture': ['festival', 'city', 'etc'],
            'experience': ['uniquetravel', 'walking', 'culture', 'etc'],
            'park': ['nature', 'city', 'etc'],
            'shopping': ['ocean', 'city', 'festival', 'etc'],
            'festival': ['culture', 'shopping', 'etc'],
            'walking': ['uniquetravel', 'mountain', 'park', 'nature', 'culture', 'history', 'city'],
            'etc': ['nature', 'walking', 'park', 'shopping', 'uniquetravel', 'culture', 'festival', 'experience',
                    'history', 'mountain', 'ocean', 'city']
        }

        # 한글 테마를 영어로 치환하기 위한 매핑 사전
        theme_mapping = {
            '자연': 'nature',
            '도시': 'city',
            '역사': 'history',
            '이색여행': 'uniquetravel',
            '바다': 'ocean',
            '산': 'mountain',
            '문화': 'culture',
            '체험': 'experience',
            '공원': 'park',
            '쇼핑': 'shopping',
            '축제': 'festival',
            '걷기': 'walking',
            '기타': 'etc'
        }

        # condition['theme']가 단일 값일 때
        main_theme = theme_mapping.get(condition['theme'], 'etc')
        related_themes = theme_groups.get(main_theme, [])

        # Theme 가중치 계산
        def calculate_theme_weight(row):
            if row[main_theme] == 1:
                return 0.5
            elif any(row[theme] == 1 for theme in related_themes):
                return 0.35
            else:
                return 0.15

        data['ThemeWeight'] = data.apply(calculate_theme_weight, axis=1)

        # 연령대 점수와 가중치 곱셈
        data['FinalScore'] = data[user_age_group] * data['CompanionWeight'] * data['ThemeWeight']

        # 최종 점수 기준 상위 12K개 여행지 추천
        recommended = data.sort_values(by='FinalScore', ascending=False).head(k * 12)

        # 이름, 위도, 경도 리스트로 반환
        result = list(zip(recommended['placename'], recommended['latitude'], recommended['longitude']))

        # 이름과 상세설명만 반환
        return result

    response = recommend_travel_destinations(data, keywords, user_age, k)

    response = state['playing_context'].append(response)

    return state


def make_schedule(state):
    model = ChatOpenAI(temperature=0.3, model="gpt-4-turbo")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Your task is to generate a travel itinerary based on the provided data and specific requirements. The output should include
        a daily schedule in JSON format.

        ====
        def generate_itinerary(hotels: list, tourist_spots: list, restaurants: dict, days: int) -> dict:
            Generates a travel itinerary for 'n' days using the provided data. The itinerary should be returned in JSON format
            and follow these guidelines:

            - Each day's itinerary must include breakfast, lunch, and dinner. Breakfast and lunch should be at different restaurants, 
              while dinner can also be at a restaurant or bar.
            - Each day should include visits to 2 different tourist spots, with a balanced mix of activities (e.g., cultural, 
              natural, historical).
            - A different hotel should be assigned each night, with a maximum of 3 consecutive nights at the same hotel to ensure variety.
            - The itinerary should start at a central location on the first day and end at the same location on the last day 
              (e.g., a train station), but this location is not considered a tourist spot.
            - The itinerary should be unique and slightly different each time it is generated, with random selections of restaurants, 
              tourist spots, and hotels to provide a varied experience.
            - The itinerary should ensure that daily movements are geographically logical:
                - Each day's activities should occur within a reasonable proximity, ideally within 5 km of each other.
                - The itinerary should follow a general direction or loop, minimizing unnecessary backtracking.
            
            Args:
                hotels (list): A list of tuples, where each tuple contains the hotel name, latitude, and longitude 
                               (e.g., [(name, latitude, longitude), ...]).
                tourist_spots (list): A list of tuples, where each tuple contains the tourist spot name, latitude, and longitude 
                                      (e.g., [(name, latitude, longitude), ...]).
                restaurants (dict): A dictionary with keys '아침', '점심', '저녁', and values are lists of tuples containing the 
                                    restaurant name, latitude, and longitude (e.g., {{'아침': [(name, latitude, longitude), ...], 
                                    '점심': [...], '저녁': [...]}}).
                days (int): The number of days for the itinerary.

            Returns:
                dict: A JSON-formatted dictionary representing the travel itinerary. Each day's itinerary should include locations 
                      for breakfast, lunch, dinner, and two tourist spots, along with their respective names and coordinates.
        ====
        """),
        ("human", "{hotels}, {tourist_spots}, {restaurants}, {days}")
    ])

    chain = prompt | model

    result = chain.invoke({
        "hotels": state['hotel_context'], 
        "tourist_spots": state['playing_context'], 
        "restaurants": state['foods_context'], 
        "days": int(state['keywords']['days'])  # days는 int로 변환
    })

    schedule = result.content

    # JSON 부분 찾기
    json_start = schedule.find('{')
    json_end = schedule.rfind('}')
    json_data_str = schedule[json_start:json_end+1]

    # JSON 파싱
    itinerary_data = json.loads(json_data_str)

    # 상태 업데이트
    schedule = json.dumps(itinerary_data['itinerary'], indent=4, ensure_ascii=False)
    state['scheduler'] = schedule
    state['model'] = model
    state['is_valid'] = 1

    return state


def validation(state):
    model = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Your task is to classify the sentence into one of the following categories: ["Other", "Again", "Good"]
        ====
        def classify(sentence: str) -> Literal["Other", "Again", "Good"]:
            Classify the sentence into one of the following categories based on the user's intent:

            - Again: The user wants to start over or re-enter information. Examples include phrases like '처음부터 할래' or '다시 입력하게 해줘'.
            - Other: The user wants to search again or is unsatisfied with the current result. Examples include phrases like '다시 찾아줘' or '여기 별로야'.
            - Good: The user is satisfied or wants to move forward. Examples include phrases like '알겠어' or '좋아'.

            Args:
                sentence (str): The sentence that needs to be classified.

            Returns:
                Literal["Other", "Again", "Good"]: The classification of the sentence.
        ====
        # Get result
        result = classify_sentence(sentence)
        return result

        ====
        # Response format
        classify("{human}")
        """),
        ("human", "{human}")
    ])
    chain = prompt | model
    answer = chain.invoke({"human": state['second_sentence']})
    answer = answer.content
    state['second_sentence'] = answer
    state['model'] = model
    return state