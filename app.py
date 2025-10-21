import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma # <-- DEĞİŞTİ
from langchain_community.document_loaders import TextLoader # <-- DEĞİŞTİ
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate # <-- DEĞİŞTİ
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Ortam değişkenlerini yükle (.env dosyasından) ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY .env dosyasında bulunamadı. Lütfen API anahtarını ekle.")
    st.stop()

# --- Streamlit sayfa başlığı ---
st.set_page_config(page_title="Discover Türkiye Chatbot", page_icon="🇹🇷")
st.title("🇹🇷 Discover Türkiye Chatbot")
st.write("Türkiye’deki şehirleri keşfedin! Bana sorular sorun, sohbet edelim.")

# --- Veri Yükleme ve İşleme  ---
# Bu fonksiyon, verilerin olduğu dosyadan verileri alır
# Uzun metinleri 1000 karakterlik parçalara böler 
# Her parçayı “embedding” haline getirip Chroma veritabanına ekler
# Bu işlem cache'lenir, böylece uygulama her yeniden çalıştığında tekrarlanmaz
@st.cache_resource
def load_and_process_data():
    loader = TextLoader("turkiye_turizm.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)  
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vectordb = Chroma.from_documents(docs, embeddings)
    return vectordb

vectordb = load_and_process_data() #Embedding’lenmiş verileri içerir

# --- Prompt Şablonları ---
# Bu bölümde modelin nasıl düşüneceğini ve cevap vereceğini tanımlayan iki ayrı "prompt şablonu" oluşturuluyor
# Amaç, modeli iki aşamalı düşündürmektir:
# CONDENSE_QUESTION_PROMPT → Kullanıcının takip sorularını geçmiş bağlama göre yeniden yazar (soruyu anlamlı hale getirir)
# QA_PROMPT → Modelin cevabı hangi üslupta, hangi kaynaklara dayanarak vereceğini belirler
# Böylece model hem "neyi sorduğunu" hem de "nasıl cevaplayacağını" ayrı ayrı öğrenir

# 1. Soru Yoğunlaştırma Prompt'u
_template = """
Aşağıdaki sohbet geçmişi ve takip sorusu verildiğinde, takip sorusunu, sohbet geçmişindeki şehir adı gibi ana konuyu MUTLAKA içerecek şekilde, tek başına bir soru olarak yeniden ifade et.

Örnek:
Sohbet Geçmişi:
Kullanıcı: Konya'da ne yenir?
Yapay Zeka: Konya'da etli ekmek yiyebilirsiniz.
Takip Sorusu: peki orada nereler gezilir?
Tek Başına Soru: Konya'da nereler gezilir?

Sohbet Geçmişi:
{chat_history}
Takip Sorusu: {question}
Tek Başına Soru:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# 2. Cevaplama Prompt'u
QA_PROMPT_TEMPLATE = """
Sen Türkiye hakkında her şeyi bilen, arkadaş canlısı ve yardımsever bir turizm rehberisin.
Sana verilen metin parçalarını ({context}) kullanarak son soruyu ({question}) cevapla.
Cevaplarını SADECE verilen metindeki bilgilere dayanarak oluştur.
Bilgileri birleştirerek, tekrar etmeden, akıcı bir paragraf halinde sun.
Cevabın sanki bir rehberle sohbet ediyormuş gibi sıcak ve doğal olsun. Eğer bilgi metinde yoksa veya soru konu dışıysa, "Bu konuda elimde bilgi yok, ama istersen Türkiye'deki diğer şehirlerle ilgili önerilerde bulunabilirim." de.

Yardımsever Cevabın:
"""
QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# --- Dil Modeli (LLM), Hafıza ve Sohbet Zinciri ---
# Burada LangChain'e hangi yapay zekâ modelini (LLM) kullanacağını söylüyoruz
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Kullanılan Gemini sürümünü belirler
    google_api_key=google_api_key, # API erişim anahtarını iletir
    temperature=0.7, # Modelin yaratıcılığını ayarlar
    convert_system_message_to_human=True # LangChain–Streamlit uyumunu sağlar
)

# --- Hafıza, Bilgi Arama ve Sohbet Zinciri ---
# Bu bölüm chatbot'un sohbet geçmişini hatırlamasını, veritabanında arama yapmasını ve cevap üretmesini sağlar

# Hafıza (Memory):
# - st.session_state.memory → Konuşma geçmişi oturum bazlı olarak saklanır
# - ConversationBufferMemory → Önceki mesajları sıralı şekilde tutar (bağlam korunur)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Retriever:
# - vectordb.as_retriever(k=4) → Chroma veritabanında en alakalı 4 belgeyi getirir
# - Bu belgeler embedding benzerliğine göre seçilir
retriever = vectordb.as_retriever(search_kwargs={'k': 4})

# ConversationalRetrievalChain:
# - LLM (Gemini) + Retriever + Memory birleşimiyle çalışır
# - condense_question_prompt → Takip sorularını bağlama göre yeniden yazar
# - QA_PROMPT → Belgelerden gelen bilgiyi rehber üslubuyla kullanıcıya sunar
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
)

# --- Streamlit Sohbet Arayüzü ---
# Bu kısım kullanıcı ile chatbot arasındaki görsel sohbet deneyimini oluşturur.

# Geçmiş Mesajlar:
# - st.session_state.messages → Oturum boyunca mesaj geçmişini saklar.
# - Eğer yoksa boş liste oluşturulur.
# - Daha önceki tüm konuşmalar for döngüsüyle ekranda görüntülenir.
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı Girdisi:
# - st.chat_input() → Kullanıcının mesaj yazdığı alan.
# - Kullanıcı mesaj gönderdiğinde hem ekrana yazılır hem geçmişe eklenir.
if prompt := st.chat_input("İstanbul hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Chatbot Cevabı:
# - st.spinner() → "Düşünüyorum..." animasyonu gösterir.
# - conversation_chain() → Model, veritabanı ve hafızayı kullanarak cevabı üretir.
# - Cevap ekranda gösterilir ve geçmişe kaydedilir.
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            try:
                result = conversation_chain({"question": prompt})
                response = result["answer"]
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"❌ Bir hata oluştu: {str(e)}")




