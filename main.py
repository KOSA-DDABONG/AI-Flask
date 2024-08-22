from flask import Flask, request, jsonify, render_template
import gptlogic

app = Flask(__name__)

this_state = gptlogic.input_text()

def schedule_make_graph(this_state):
    this_state = gptlogic.foods_search(this_state)
    this_state = gptlogic.hotels_search(this_state)
    this_state = gptlogic.places_search(this_state)
    this_state = gptlogic.make_schedule(this_state)
    return this_state['scheduler']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/making', methods=['POST'])
def making_schedule():
    
    # 입력된 질문 가져오기
    data = request.json
    question = data.get('question', '')
    user_age = request.args.get("userAge")

    # 상태 업데이트
    this_state['question'] = question
    this_state['user_age'] = user_age

    # 키워드 추출 함수 호출
    keywords, response = gptlogic.find_keywords(this_state)

    # 업데이트된 키워드를 this_state에 반영
    this_state['keywords'] = keywords

    # 상태에 따른 응답
    if response != 'End':
        return jsonify({'response': response})
    else:
        schedule = schedule_make_graph(this_state)
        return jsonify({'response': schedule})
    
@app.route('/validating', methods=['POST'])
def validate():
    data = request.json

    question = data.get('question', '')
    this_state['second_sentence'] = question

    this_state = gptlogic.validation(this_state)
    if this_state['second_sentence']=='Good':
        return jsonify({'response': "일정이 생성되었습니다."})
    
    elif this_state['second_sentence']=='Other':
        this_state = gptlogic.make_schedule(this_state)
        return jsonify({'response': this_state['scheduler']})
    
    elif this_state['second_sentence']=='Again':
        this_state = gptlogic.input_text()
        return this_state
        # 다시 making_schedule 로직실행

if __name__ == '__main__':
    app.run(debug=True)