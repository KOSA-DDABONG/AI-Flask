from flask import Flask, request, jsonify, render_template, Response
import gptlogic
import json  # 여기서 json 모듈을 import 합니다.

app = Flask(__name__)


def schedule_make_graph(this_state):
    this_state = gptlogic.searching(this_state, 'food')
    this_state = gptlogic.searching(this_state, 'hotel')
    this_state = gptlogic.searching(this_state, 'place')
    this_state = gptlogic.make_schedule(this_state)

    return this_state


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_user', methods=['POST'])
def get_user():
    user = gptlogic.UserState()
    state = user.get_state()
    state['user_age'] = request.args.get("userAge")
    state['user_token'] = request.args.get("userToken")
    
    serializable_state = {k: v for k, v in state.items()}
    
    response_json = json.dumps(serializable_state, ensure_ascii=False)  # UTF-8 인코딩
    return Response(response_json, content_type="application/json; charset=utf-8")


@app.route('/making', methods=['POST'])
def making_schedule():
    # 입력된 질문 가져오기
    data = request.json
    
    state = data
    
    # 키워드 추출 함수 호출
    keywords, response = gptlogic.find_keywords(state)

    # 업데이트된 키워드를 this_state에 반영
    state['keywords'] = keywords
    state['response'] = response

    serializable_state = {k: v for k, v in state.items()}
    response_json = json.dumps(serializable_state, ensure_ascii=False)  # UTF-8 인코딩
    
    # 상태에 따른 응답
    state = schedule_make_graph(state)
    serializable_state = {k: v for k, v in state.items()}
    response_json = json.dumps(serializable_state, ensure_ascii=False)  # UTF-8 인코딩
        
    return Response(response_json, content_type="application/json; charset=utf-8")
    

@app.route('/validating', methods=['POST'])
def validate():
    data = request.json

    state = data

    state = gptlogic.validation(state)
    if state['second_sentence'] == 'Good':
        response_json = json.dumps({'response': "일정이 생성되었습니다."}, ensure_ascii=False)
        return Response(response_json, content_type="application/json; charset=utf-8")

    elif state['second_sentence'] == 'Other':
        state = gptlogic.make_schedule(state)
        serializable_state = {k: v for k, v in state.items()}
        response_json = json.dumps(serializable_state, ensure_ascii=False)  # UTF-8 인코딩
        print("다시 만들어주는 로직")
        return Response(response_json, content_type="application/json; charset=utf-8")

    elif state['second_sentence'] == 'Again':
        new_state = gptlogic.UserState().get_state()
        new_state['user_token'] = state['user_token']
        new_state['user_age'] = state['user_age']
        print("첨부터의 로직")
        serializable_state = {k: v for k, v in new_state.items()}
        response_json = json.dumps(serializable_state, ensure_ascii=False)  # UTF-8 인코딩
        return Response(response_json, content_type="application/json; charset=utf-8")
        # 다시 making_schedule 로직실행


@app.route('/updating_place', methods=['POST'])
def updating():
    data = request.json

    state = data.get('state', '')
    change_place = data.get('placename', '')
    result_place = gptlogic.update_place(change_place, state)
    return jsonify({'result': result_place})



if __name__ == '__main__':
    app.run(debug=True)