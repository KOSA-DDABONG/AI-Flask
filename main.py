from flask import Flask, request, jsonify, render_template
import gptlogic

app = Flask(__name__)


def schedule_make_graph(this_state):
    this_state = gptlogic.foods_search(this_state)
    this_state = gptlogic.hotels_search(this_state)
    this_state = gptlogic.places_search(this_state)
    this_state = gptlogic.make_schedule(this_state)
    return this_state['scheduler']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_user', methods=['POST'])
def get_user():
    user = gptlogic.UserState()
    state = user.get_state()
    state['user_age'] = request.args.get("userAge")
    state['user_token'] = request.args.get("userToken")
    print(state)

    serializable_state = {k: v for k, v in state.items() if k != 'model'}
    print(serializable_state)
    return jsonify(serializable_state)


@app.route('/making', methods=['POST'])
def making_schedule():
    # 입력된 질문 가져오기
    data = request.json
    print("일정 생성")
    print("[data]")
    print(data)
    state = data
    print("[state]")
    print(state)

    # 키워드 추출 함수 호출
    keywords, response = gptlogic.find_keywords(state)

    # 업데이트된 키워드를 this_state에 반영
    state['keywords'] = keywords
    state['response'] = response

    serializable_state = {k: v for k, v in state.items() if k != 'model'}
    print(serializable_state)
    # 상태에 따른 응답
    if response != 'End':
        #return jsonify({'response': response})
        return jsonify(serializable_state)
    else:
        schedule = schedule_make_graph(state)
        #return jsonify({'response': schedule})
        return jsonify(serializable_state)


@app.route('/validating', methods=['POST'])
def validate():
    data = request.json

    state = data.get('question', '')

    state = gptlogic.validation(state)
    if state['second_sentence'] == 'Good':
        return jsonify({'response': "일정이 생성되었습니다."})

    elif state['second_sentence'] == 'Other':
        state = gptlogic.make_schedule(state)
        return jsonify({'response': state['scheduler']})

    elif state['second_sentence'] == 'Again':
        new_state = gptlogic.UserState().get_state()
        new_state['user_token'] = state['user_token']
        new_state['user_age'] = state['user_age']
        return new_state
        # 다시 making_schedule 로직실행


if __name__ == '__main__':
    app.run(debug=True)