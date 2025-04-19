import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None

def get_chatbot_response(user_input):
    
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    
    bot_input_ids = new_user_input_ids if st.session_state.chat_history_ids is None else torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)

   
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    bot_output = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_output


st.title("Chat with Chatbot")
st.markdown("Ask me anything and I'll do my best to answer!")


user_input = st.text_input("You: ", "")

if user_input:
    bot_response = get_chatbot_response(user_input)
    st.text_area("Chatbot:", value=bot_response, height=300)

if 'history' not in st.session_state:
    st.session_state.history = []

if user_input and bot_response:
    st.session_state.history.append(f"You: {user_input}")
    st.session_state.history.append(f"Chatbot: {bot_response}")

for message in st.session_state.history:
    st.write(message)

st.markdown("Type your questions and interact with the chatbot. For example, you can ask about any topic!")

