def analyze_message(message):
    message = message.lower()

    if "headache" in message:
        return "It sounds like you have a headache. Make sure you rest and stay hydrated."
    elif "fever" in message:
        return "You mentioned a fever. Please monitor your temperature and drink plenty of fluids."
    elif "cough" in message:
        return "For a cough, try to rest your throat and stay hydrated. If it worsens, consult a doctor."
    else:
        return "I'm not sure about your symptoms. Please provide more details or consult a medical professional."
