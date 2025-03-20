import streamlit as st
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
from peft import PeftModel

# Define model directory and base model
model_dir_legal_peft = "legal-led-summary-training-1742381354/checkpoint-350"
base_model_name = "nsi319/legal-led-base-16384"

# Load tokenizer and model
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    # Load tokenizer from the local directory
    tokenizer = LEDTokenizer.from_pretrained(model_dir_legal_peft)
    
    # Load base model
    base_model = LEDForConditionalGeneration.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    
    # Load adapter weights
    device = "cpu"
    model = PeftModel.from_pretrained(base_model, model_dir_legal_peft, torch_dtype=torch.bfloat16, is_trainable=False)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Function to summarize text
def summarize_text(text, max_length=508):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=16048)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Generate summary
    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True
    )
    
    # Decode and return summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("Legal Document Summarization Chatbot")

# Chatbot UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter legal text for summarization...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate summary
    with st.chat_message("assistant"):
        summary = summarize_text(user_input)
        st.markdown(summary)
    
    st.session_state.messages.append({"role": "assistant", "content": summary})