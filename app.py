import streamlit as st
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

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

    device = "cpu"  # or "cuda" if available
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer_summary, model_summary, device_summary = load_summary_model()

# ------------------ LOAD CHAT MODEL ------------------ #
@st.cache_resource
def load_qa_model():
    qa_model_name = "meta-llama/Llama-4-Maverick-17B-128E-Original"

    tokenizer = AutoTokenizer.from_pretrained(
        qa_model_name,
        use_auth_token=True  # required if it's gated
    )

    model = AutoModelForCausalLM.from_pretrained(
        qa_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # let Transformers place the model on the appropriate device
        use_auth_token=True
    )

    device = model.device  # get assigned device from auto-mapping

    model.eval()
    return tokenizer, model, device

# ------------------ SUMMARIZATION FUNCTION ------------------ #
def summarize_text(text, max_length=508):
    prompt = (
        "Summarize the following privacy policy with the following structure:\n\n"
        "**TL;DR:** A concise summary in 2-3 sentences.\n\n"
        "**Detailed Summary:**\n"
        "- **Introduction:** Briefly introduce the company’s stance on privacy.\n"
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
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer_qa(prompt, return_tensors="pt", return_token_type_ids=False).to(device_qa)
    outputs = model_qa.generate(**inputs, max_new_tokens=200)
    return tokenizer_qa.decode(outputs[0], skip_special_tokens=True)

# ------------------ CHAT UI ------------------ #
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter your input here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        if mode == "Summarize Policy":
            summary = summarize_text(user_input)
            st.session_state.last_summary = summary  # Save for future Q&A
            st.markdown(summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})

        elif mode == "Ask a Question":
            context = st.session_state.last_summary or "No summary available yet. Please summarize a policy first."
            answer = answer_question(user_input, context=context)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
