import streamlit as st
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration, pipeline
from peft import PeftModel
import os
from huggingface_hub import login

# ------------------ HUGGING FACE LOGIN ------------------ #
hf_token = st.secrets.get("HUGGINGFACE_HUB", os.environ.get("HUGGINGFACE_HUB"))
if hf_token:
    login(token=hf_token)
else:
    st.warning("Hugging Face token not found. Please set HUGGINGFACE_HUB in secrets or environment variables.")

# ------------------ PAGE SETUP ------------------ #
st.set_page_config(page_title="Genetic Privacy Policy Chatbot", layout="centered")
st.title("🧬 Privacy Policies Summarization Chatbot")

st.markdown("""
This is a chatbot that helps you to summarize privacy policies of genetic testing companies.  
You can just copy the text in the privacy statement and paste it down below.  
I will create a concise summary for you and you can ask some questions if you like.
""")

# ------------------ SIDEBAR MODE SWITCH ------------------ #
st.sidebar.title("🛠️ Options")
mode = st.sidebar.radio("Choose mode:", ["Summarize Policy", "Ask a Question"])

# ------------------ STATE INITIALIZATION ------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""

# ------------------ LOAD SUMMARIZATION MODEL ------------------ #
@st.cache_resource
def load_summary_model():
    model_dir_legal_peft = "legal_led_for_summary_bigger_model"
    base_model_name = "nsi319/legal-led-base-16384"

    tokenizer = LEDTokenizer.from_pretrained(model_dir_legal_peft)
    base_model = LEDForConditionalGeneration.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, model_dir_legal_peft, torch_dtype=torch.bfloat16, is_trainable=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer_summary, model_summary, device_summary = load_summary_model()

# ------------------ LOAD CHAT PIPELINE ------------------ #
@st.cache_resource
def load_qa_pipeline():
    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    
    try:
        qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_auth_token=True
        )
        return qa_pipeline
    except Exception as e:
        st.error(f"Failed to load Q&A pipeline: {str(e)}")
        st.error("Please ensure you have access to the model and your Hugging Face token is valid.")
        return None

qa_pipeline = load_qa_pipeline()

# ------------------ SUMMARIZATION FUNCTION ------------------ #
def summarize_text(text, max_length=1016):
    prompt = (
        "Summarize the following privacy policy with the following structure:\n\n"
        "**TL;DR:** A concise summary in 2-3 sentences.\n\n"
        "**Detailed Summary:**\n"
        "- **Introduction:** Briefly introduce the company's stance on privacy.\n"
        "- **Data Collection:** Outline what personal data is collected.\n"
        "- **Data Usage:** Explain how the collected data is used.\n"
        "- **Data Sharing:** Describe who the data is shared with and under what conditions.\n"
        "- **User Controls:** Explain how users can manage their data.\n"
        "- **Legal Considerations:** Mention compliance with laws and any legal obligations.\n"
        "- **Important Notes:** List key points regarding user rights and protections.\n\n"
        "Privacy Policy:\n"
        f"{text}"
    )

    inputs = tokenizer_summary(prompt, return_tensors="pt", truncation=True, max_length=16048)
    inputs = {key: val.to(device_summary) for key, val in inputs.items()}

    summary_ids = model_summary.generate(
        input_ids=inputs["input_ids"],
        max_length=max_length,
        num_beams=7,
        no_repeat_ngram_size=3,
        early_stopping=True,
        temperature=0.5,
        top_p=0.8,
        top_k=40,
        do_sample=True
    )

    return tokenizer_summary.decode(summary_ids[0], skip_special_tokens=True)

# ------------------ Q&A FUNCTION ------------------ #
def answer_question(question, context):
    if qa_pipeline is None:
        return "Q&A functionality is currently unavailable. Please try again later."
    
    prompt = f"""<|im_start|>system
You are a helpful assistant that answers questions about privacy policies.<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
    
    try:
        response = qa_pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=2  # Llama's eos token
        )
        
        # Extract just the assistant's response
        full_response = response[0]['generated_text']
        assistant_response = full_response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        return assistant_response.strip()
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating the answer."

# ------------------ CHAT UI ------------------ #
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter your input here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        if mode == "Summarize Policy":
            with st.spinner("Generating summary..."):
                try:
                    summary = summarize_text(user_input)
                    st.session_state.last_summary = summary
                    st.markdown(summary)
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't generate a summary."})

        elif mode == "Ask a Question":
            if not st.session_state.last_summary:
                st.warning("Please summarize a policy first before asking questions.")
                st.session_state.messages.append({"role": "assistant", "content": "Please summarize a policy first before asking questions."})
            else:
                with st.spinner("Generating answer..."):
                    answer = answer_question(user_input, context=st.session_state.last_summary)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})