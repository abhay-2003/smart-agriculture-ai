import google.generativeai as genai

class AgricultureChatbot:
    def __init__(self, api_key):
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None

    def get_response(self, user_prompt, chat_history):
        if not self.model:
            return "API Key not configured."
            
        system_instruction = (
            "You are an expert agriculture AI assistant helping farmers. "
            "Only answer questions related to agriculture, farming, crops, fertilizers, "
            "soil health, plant diseases, irrigation, and weather. If the user asks about "
            "anything else, politely decline and steer the conversation back to agriculture."
        )
        
        full_prompt = f"System Instruction: {system_instruction}\n\n"
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
            
        full_prompt += f"User: {user_prompt}\nAssistant:"
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with AI: {e}"
