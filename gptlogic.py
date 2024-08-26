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
        }

    def get_state(self):
        return self.state


def get_db_connection():
    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT", 3306))
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")

    return create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')


def get_db(kind):
    # SQLAlchemy 엔진 생성
    engine = get_db_connection()

    queries = {
        "food": """
                SELECT placename, latitude, longitude, teenager, twenties, thirties, fourties, fifties, sixties, 
                       foodtype, bar, parking, operatinghours0, operatinghours1, operatinghours2, operatinghours3, operatinghours4, operatinghours5
                FROM PLACE P JOIN RESTAURANT R ON P.placeid = R.placeid
              """,
        "hotel": """
                SELECT placename, latitude, longitude, teenager, twenties, thirties, fourties, fifties, sixties, 
                       parking, hoteltype
                FROM PLACE P JOIN HOTEL R ON P.placeid = R.placeid
              """,
        "place": """
                SELECT placename, latitude, longitude, teenager, twenties, thirties, fourties, fifties, sixties, 
                       single, couple, parents, family, friends, experience, park, uniquetravel, nature, 
                       culture, festival, shopping, history, walking, city, etc, ocean, mountain
                FROM PLACE P JOIN TOUR R ON P.placeid = R.placeid
              """
    }

    if kind in queries:
        df = pd.read_sql(queries[kind], engine)
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
    model = ChatOpenAI(temperature=0.15, model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

    # GPT에게 현재 입력된 정보로부터 키워드 추출 요청
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
    You need to extract the most relevant information from the user's sentence in the context of a trip. 
    The user may provide information about the following categories:
    1. How many days they want to travel (days).
    2. What transportation they have (transport).
    3. Who they are traveling with (companion).
    4. What theme they prefer for the trip (theme).
    5. What type of food they prefer (food).

    Basically, you find keywords from user's input, but you can also recommend keyword as possible as you can.
    For example
    - if the input is '쉬러 가고 싶어' means '휴식' so you can recommend '바다' or '자연' or '공원'
    - if the input is '라면먹고 싶어' means wanna eat '라면' so you can recommend '한식' or '일식'
    - if the input is '사람들을 만나러 가고싶어' means wanna hang with people, so you can recommend '도시' or '체험' or '쇼핑'
    - if the input is '아무거나' or '상관없어' or '아무데나' means everywhere and everything, so you can recommend randomly from format.
    - if the input is '나머지는 추천해줘' or '나머지는 알아서 해줘' means recommend the rest of categories, so you can recommend from format for the rest of categories.

    Following Rules:
    - if you are short of information from user's sentence, but user just want recommend of rest, then you can recommend from format.
    - Recommending is your option, but you just return one keyword at one category.

    days
    - just return Number of days. liks 'N'. just integer.

    transport
    - if user have a car, or rent a car means '자차', but if user use taxi, bus, airplane, or other things not a car, then '대중교통'

    response
    - When the number of categories provided from the user's input is insufficient, a message providing additional information may be transmitted to the user as response.
    -  Your response will respond appropriately to the other person's sentences as if they were having a real conversation and return questions about the missing keywords.
    - React so that users can feel that they are talking.
    - response like kind and natural tour guide. 
    - if you get all categories, then just return 'End' at response.

   theme
   - you have to select keywords in the following format.
   - If the keyword you extracted is not in the format, you should return the most similar list of the list presented in the format.


    user input: "{question}, {keywords}"

    Return the information in the following format:
    {{
        'days': 'Number of days' or None,
        'transport': '자차/대중교통' or None,
        'companion': '가족/부모/친구/연인/혼자' or None,
        'theme': '자연/걷기/쇼핑/공원/이색여행/문화/체험/역사/산/바다/도시' or None,
        'food': '한식/양식/중식/일식/아시아' or None,
        'response': 'response'
    }}
    """),
        ("human","{question}, {keywords}")
    ])

    chain = prompt | model
    response = chain.invoke({"question": question, "keywords": keywords})
    extracted_keywords = ast.literal_eval(response.content)

    # 이미 대답된 정보를 저장
    for key, value in extracted_keywords.items():
        if key in keywords and value is not None and key != 'response':
            if keywords[key] is None:
                keywords[key] = value

    return keywords, extracted_keywords['response']

def recommend_restaurants(data, condition, user_age, k):
        # 연령대 설정
        user_age_group = 'sixties' if condition['companion'] == '부모' else  get_age_group(user_age)

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

def recommend_hotels(data, condition, user_age, k):
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

def recommend_places(data, condition, user_age, k):
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

def searching(state, whatwant):
   
    if whatwant == 'food':
        user_age = int(state['user_age'])
        data = get_db('food')

        if state['foods_context'] is None:
            state['foods_context'] = []  # None이 아니라 빈 리스트로 초기화

        keywords = state['keywords']
        k = int(state['keywords']['days'])

        response = recommend_restaurants(data, keywords, user_age, k)

        response = state['foods_context'].append(response)

        return state

    elif whatwant == 'place':
        user_age = int(state['user_age'])
        data = get_db('place')

        if state['playing_context'] is None:
            state['playing_context'] = []  # None이 아니라 빈 리스트로 초기화

        keywords = state['keywords']
        k = int(state['keywords']['days'])

        response = recommend_places(data, keywords, user_age, k)

        response = state['playing_context'].append(response)

        return state

    elif whatwant == 'hotel':
        user_age = int(state['user_age'])
        data = get_db('hotel')

        if state['hotel_context'] is None:
            state['hotel_context'] = []  # None이 아니라 빈 리스트로 초기화

        keywords = state['keywords']
        k = int(state['keywords']['days'])

        response = recommend_hotels(data, keywords, user_age, k)

        response = state['hotel_context'].append(response)

        return state

def make_schedule(state):
    model = ChatOpenAI(temperature=0.33, model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        Your task is to generate a travel itinerary based on the provided data and specific requirements. The output should include both the reasoning behind the itinerary and a daily schedule in JSON format.

        ====
        def generate_itinerary(hotels: list, tourist_spots: list, restaurants: dict, days: int) -> dict:
            Generates a travel itinerary for 'n' days using the provided data. The itinerary should be returned in JSON format and follow these guidelines:

            - Each day's itinerary must include breakfast, lunch, and dinner, but the timing of meals should vary to simulate real-life travel flexibility.
            - Tourist spots should be selected in a way that ensures a variety of experiences, avoiding repetition and ensuring a mix of cultural, natural, and historical sites.
            - Each day can include 1 to 3 Tourist spots.
            - Randomize the choice of hotels, ensuring that no more than 3 consecutive nights are spent in the same hotel, but not necessarily to choice new at every night.
            - Ensure a logical flow of travel that minimizes unnecessary backtracking and maximizes time spent enjoying the locations(consider latitude and longitude).
            - The itinerary should begin and end at a central location, but this location does not count as a tourist spot.
            - The JSON output should use the day number as the key and include the itinerary for that day as the value.
            - The last day should not include a hotel stay.
            - Provide a detailed explanation of why the itinerary was planned in this way, considering factors such as travel convenience, variety, user preferences, and the optimal use of time.

            Args:
                hotels (list): A list of tuples, where each tuple contains the hotel name, latitude, and longitude (e.g., [(name, latitude, longitude), ...]).
                tourist_spots (list): A list of tuples, where each tuple contains the tourist spot name, latitude, and longitude (e.g., [(name, latitude, longitude), ...]).
                restaurants (dict): A dictionary with keys '아침', '점심', '저녁', and values are lists of tuples containing the restaurant name, latitude, and longitude (e.g., {{'아침': [(name, latitude, longitude), ...], '점심': [...], '저녁': [...]}}).
                days (int): The number of days for the itinerary.

            Returns:
                dict: A dictionary with two keys:
                    1. "reasoning": A detailed explanation of the rationale behind the itinerary to korean.
                    2. "itinerary": - A JSON-formatted dictionary representing the travel itinerary. Each day's itinerary should include locations for breakfast, lunch, dinner, and tourist spots, along with their respective names and coordinates.
                                    - About Day key is not like 'Day 1' or 'day1'. just like '1'(Only Integer)

        # Example usage
        result = generate_itinerary(hotels, tourist_spots, restaurants, days)
        return result

        ====
        # Response format
        generate_itinerary({hotels}, {tourist_spots}, {restaurants}, {days})
        ====
        """),
        ("human", "{hotels}, {tourist_spots}, {restaurants}, {days}")
    ])

    chain = prompt | model

    # 올바른 변수명을 사용하여 chain을 실행합니다.
    result = chain.invoke({
        "hotels": state['hotel_context'],
        "tourist_spots": state['playing_context'],
        "restaurants": state['foods_context'],
        "days": int(state['keywords']['days'])
    })

    schedule = result.content

    # 파싱 전, 중간의 ```json```과 ```를 제거합니다.
    json_start = schedule.find('{')
    json_end = schedule.rfind('}')
    json_data_str = schedule[json_start:json_end + 1]

    # JSON을 파싱합니다.
    itinerary_data = json.loads(json_data_str)

    schedule = json.dumps(itinerary_data['itinerary'], indent=4, ensure_ascii=False)
    state['scheduler'] = schedule
    state['explain'] = itinerary_data['reasoning']
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
    return state

def place_filtering(placename, state):
    engine = get_db_connection()

    query = """
            SELECT placetype, latitude, longitude
            FROM PLACE
            WHERE placename = :placename
          """
    
    placename_info = pd.read_sql(query, engine, params={"placename": placename})

    # 기존장소 담기
    placetype = placename_info.iloc[0]['placetype']
    placename_lat = placename_info.iloc[0]['latitude']
    placename_lon = placename_info.iloc[0]['longitude']

    extracted_names = []

    if placetype == 0: #관광
        data = get_db('place')
        data = data[data['placename'] != placename]

        for _, activities in state['schedule'].items():
            if 'tourist_spots' in activities:
                extracted_names.extend([spot['name'] for spot in activities['tourist_spots']])
        
        data = data[~data['placename'].isin(extracted_names)]
        user_age = int(state['user_age'])
        k = int(state['keywords']['days'])
        result = recommend_places(data, state['keywords'], user_age, k)

    elif placetype == 1: #숙박
        data = get_db('hotel')
        data = data[data['placename'] != placename]

        for _, activities in state['schedule'].items():
            if 'hotel' in activities and activities['hotel'] and 'name' in activities['hotel']:
                extracted_names.append(activities['hotel']['name'])
        
        data = data[~data['placename'].isin(extracted_names)]
        user_age = int(state['user_age'])
        k = int(state['keywords']['days'])
        result = recommend_hotels(data, state['keywords'], user_age, k)

    elif placetype == 2: #식당
        data = get_db('food')
        data = data[data['placename'] != placename]

        for _, activities in state['schedule'].items():
            if 'breakfast' in activities and 'name' in activities['breakfast']:
                extracted_names.append(activities['breakfast']['name'])
            if 'lunch' in activities and 'name' in activities['lunch']:
                extracted_names.append(activities['lunch']['name'])
            if 'dinner' in activities and 'name' in activities['dinner']:
                extracted_names.append(activities['dinner']['name'])
        
        data = data[~data['placename'].isin(extracted_names)]
        user_age = int(state['user_age'])
        k = int(state['keywords']['days'])
        result = recommend_restaurants(data, state['keywords'], user_age, k)

    origin_place = (placename, placename_lat, placename_lon)
    return result, origin_place

def update_place(placename, state):
    new_place, origin_place = place_filtering(placename, state)

    model = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
    Your task is to recommend a new location based on the proximity and relevance of the provided data. The output should include the reasoning behind the recommendation and the selected location in JSON format.

    ====
    def recommend_new_place(new_place: list or dict, origin_place: tuple) -> dict:
        Recommends a new location from the 'new_place' list based on proximity to the 'origin_place' and other relevant factors. The recommendation should follow these guidelines:

        - The 'origin_place' is a tuple containing the place name, latitude, and longitude (e.g., (name, latitude, longitude)).
        - The 'new_place' is a list or dictionary containing new potential locations. Each location is represented as a tuple with the name, latitude, and longitude.
        - The recommendation should prioritize proximity to the 'origin_place', selecting the closest new location.
        - Consider diversity in types of locations (e.g., mixing cultural, natural, and historical sites) when possible.
        - If 'new_place' is a dictionary, consider different categories (e.g., 'tourist_spots', 'restaurants', 'hotels') and provide a balanced recommendation.
        - The JSON output should include the selected location's name and coordinates, along with the rationale behind the choice.

        Args:
            new_place (list or dict): A list or dictionary containing potential new locations. Each location is represented as a tuple (name, latitude, longitude).
            origin_place (tuple): A tuple containing the name, latitude, and longitude of the original place.

        Returns:
            list: "recommended_place": A List-formatted list with the name, latitude, and longitude of the selected location.

    # Example usage
    result = recommend_new_place(new_place, origin_place)
    return result

    ====
    # Response format
    recommend_new_place({new_place}, {origin_place})
    ====
    """),
        ("human", "{new_place}, {origin_place}")
    ])

    chain = prompt | model
    answer = chain.invoke({"new_place": new_place,
                           'origin_place': origin_place}).content
    return answer