document.getElementById('send-button').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value;
    
    if (userInput.trim() !== "") {
        addUserMessage(userInput);
        sendToServer(userInput);
        document.getElementById('user-input').value = '';
    }
});

function addUserMessage(message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.className = 'message user-message';
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addBotMessage(message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.className = 'message bot-message';
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function sendToServer(message) {
    fetch('/find_keywords', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: message })
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            addBotMessage(data.response);
        } else {
            addBotMessage('No response from server.');
        }
    })
    .catch(error => {
        addBotMessage('Error: ' + error.message);
    });
}