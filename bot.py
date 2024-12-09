import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


import os
import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader 
from langchain.indexes import VectorstoreIndexCreator 
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
import streamlit.components.v1 as com
from langchain.llms import CTransformers 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

#Env variables:
os.environ["GOOGLE_API_KEY"] = 'AIzaSyD2vwWEsCd3cLdSxVksTF2WzM7DtfBVmio'
genai.configure(api_key='AIzaSyD2vwWEsCd3cLdSxVksTF2WzM7DtfBVmio')

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
llm = CTransformers(model = 'llama-2-7b-chat.ggmlv3.q8_0.bin', model_type = 'llama', config = {'max_new_tokens': 256, 'temperature': 0.01})



# Load the pre-trained model and dataset
model = torchvision.models.resnet50(pretrained=True)
num_foods = 3  # replace with the actual number of food categories in your dataset
model.fc = nn.Linear(model.fc.in_features, num_foods)
model.load_state_dict(torch.load('model_food_1012.h5'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

food_categories = ['apple_pie', 'baklava', 'baby_back_ribs']  # replace with your food categories list

#interface
st.title("Food Detection Application")
com.html(""" 
<style> 
         *{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body{
    width: 100%;
    min-width: 100vh;
    background-color: black;
}
#main{
    width: 100%;
    min-height: 100vh;
}
.loader{
    position: absolute;
    width: 100%;
    height: 100vh;
    background-color: rgb(226, 111, 167);
    animation: pulse 1s;
    z-index: 9;
}
.zoom{ 
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-7deg);
    font-size: 13vw;
    -webkit-text-stroke-width: 0.09vh;
    -webkit-text-fill-color: transparent;
    -webkit-text-stroke-color: azure;
}
.loader h1{
    position: absolute;
    top: 45%;
    left: 50%;
    transform: rotate(-7deg) translate(-50%, -50%);
    font-size: 8vw;
    color: rgb(0, 0, 0);
}
.line{
    position: absolute;
    left: 50%;
    top: 75%;
    height: 10px;
    width: 60%;
    background-color: rgb(0, 0, 0);
    transform: translate(-50%, -50%) rotate(-7deg);

}
.line::after{
    content: " ";
    position: absolute;
    bottom: 10%;
    height: 100%;
    animation: load 1s;


}

@keyframes load {
    0%{
        width: 0%;
        background-color: rgb(255, 0, 43);
    }
    25%{
        width: 25%;
        background-color: rgb(225, 22, 90);
    }
    50%{
        width: 50%;
        background-color: rgb(233, 78, 78);
    }
    75%{
        width: 75%;
        background-color: rgb(200, 83, 83);

    }
    100%{
        width: 100%;
        background-color: rgb(226, 111, 167);
    }
}

@keyframes pulse {
    0%{
        
        background-color: rgb(240, 182, 240);
    }
    25%{
        
        background-color: rgb(211, 165, 223);
    }
    50%{
        
        background-color: rgb(229, 157, 217);
    }
    75%{
        
        background-color: rgb(227, 159, 207);

    }
    100%{
        background-color: rgb(144, 203, 226);
    }
}
}</style>
<body>
    <div id="main">
        <div class = 'loader'>
            <h2 class="zoom">EATSMART</h2>
            <h1 class = "scramble">EATSMART</h1>
            <h2 class="line"></h2>
        </div>
    </div>


    <script src="home.js"></script>
    
</body>
</html>""")




uploaded_image = st.file_uploader("Upload an image", type="jpg")

if uploaded_image is not None:
    # Process the uploaded image
    image = Image.open(uploaded_image)
    image = transform(image).unsqueeze(0)

    # Perform food detection
    output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the index with the highest probability
    max_probability, max_index = torch.max(probabilities, 0)
    food_category = food_categories[max_index]

    # Load the calorie information from a file or database
    calories = load_calorie_information(food_category)

    st.write(f"Estimated food category: {food_category}")
    st.write(f"Estimated calories: {calories}")

    # Display the uploaded image
    plt.imshow(image[0].permute(1, 2, 0))
    st.pyplot()



#chatbot:
#to extract text from bunch of PDFs 
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


#to generate chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#vectordatabase linkage
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


#LLM chain of Google Gemini
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    if 'messages' not in st.session_state: 
            st.session_state.messages = []



    for message in st.session_state.messages: 
            st.chat_message(message['role']).markdown(message['content'])


    if user_question: 
            response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
            st.chat_message('user').markdown(user_question)
            st.session_state.messages.append({'role': 'user', 'content': user_question})

            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

    #print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.header("Chat with your virtual nutritionist!")

    user_question = st.text_input("Hey, how can I help you?")

    if user_question:
        user_input(user_question)

#loading pdfs, and linking   
pdf_docs = ['food.pdf', 'sample.pdf', 'diet.pdf', 'weightloss.pdf', 'weightgain.pdf']
raw_text = get_pdf_text(pdf_docs)
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)

#dietary plan bot?

def dietrec():
    age = st.number_input("Enter your age")
    gender = st.selectbox("Enter your gender", ("M", "F"))
    weight = st.number_input("Enter your weight")
    height = st.number_input("Enter your height")
    veg_or_nonveg = st.selectbox("select", ("veg", "non-veg"))
    disease = st.text_input("Any specific disease")
    region = st.text_input("Enter your region")
    allergics = st.text_input("Enter any allergies")
    foodtype = st.text_input("Enter your food type")
    if st.button("Click to proceed"):

        prompt_template_resto = PromptTemplate(
        input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergics', 'foodtype'],
        template="Diet Recommendation System:\n"
                "I want you to recommend 6 breakfast names, 5 dinner names, and 6 workout names, "
                "based on the following criteria:\n"
                "Person age: {age}\n"
                "Person gender: {gender}\n"
                "Person weight: {weight}\n"
                "Person height: {height}\n"
                "Person veg_or_nonveg: {veg_or_nonveg}\n"
                "Person generic disease: {disease}\n"
                "Person region: {region}\n"
                "Person allergics: {allergics}\n"
                "Person foodtype: {foodtype} in minimum 100 words."
    )
        
        results = llm(prompt_template_resto.format(age = age, gender = gender, weight = weight, height = height, veg_or_nonveg= veg_or_nonveg, disease = disease, region = region, allergics= allergics, foodtype = foodtype))
        print(results)
        st.write(results)

slider = st.sidebar.selectbox("Select", ("CHATBOT", "Dietary plan"))

if slider == "Dietary plan": 
    print(dietrec())

elif slider == "CHATBOT": 
    print(main())