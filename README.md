# LangChain ile Function Calling
> LLM'e dış fonksiyon tanıtma ve soruya göre doğru fonksiyonu otomatik seçtirme — RAG serisinin 6. adımı

[![Colab'da Aç](https://img.shields.io/badge/Colab'da%20Aç-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/yasir237/rag-langchain-6/blob/main/rag_langchain_6.ipynb)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)

---

## Problem

LLM'e "İstanbul'da hava nasıl?" dersen **tahmin eder.** Gerçek veriyi bilmez. Peki modele gerçek bir fonksiyon çağırtmak istersen ne yapacaksın?

## Çözüm

Function Calling ile LLM'e hangi fonksiyonların var olduğunu tanıtırsın. Model soruyu okur, hangi fonksiyonu hangi parametrelerle çağırması gerektiğine **kendi karar verir** ve sana söyler. Sen fonksiyonu çalıştırıp sonucu ona geri verirsin. Birden fazla fonksiyon tanıtırsan model soruya göre doğru olanı seçer — hatta aynı anda ikisini birden çağırabilir.

---

## Mimari

```
Kullanıcı sorusu
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              Function Calling Akışı                 │
│                                                     │
│  1. Pydantic BaseModel ile fonksiyon şeması tanımla │
│                        │                            │
│                        ▼                            │
│  2. llm.bind_tools([...]) ile modele tanıt          │
│                        │                            │
│                        ▼                            │
│  3. Soru gelir → model hangi fonksiyon? karar verir │
│                        │                            │
│                        ▼                            │
│  4. tool_calls → fonksiyon adı + parametreler döner │
│                        │                            │
│                        ▼                            │
│  5. Sen fonksiyonu çalıştırırsın, sonucu geri verirsin│
└─────────────────────────────────────────────────────┘
```

| Bileşen | Görevi |
|---|---|
| `BaseModel` | Fonksiyonun şemasını tanımlar — hangi parametreler, hangi tipler |
| `Field(description=...)` | Her parametreyi LLM'e açıklar — model bunu okuyarak ne dolduracağını anlar |
| `llm.bind_tools([...])` | Modele fonksiyonları tanıtır |
| `tool_calls` | Modelin döndürdüğü karar — hangi fonksiyon, hangi argümanlar |
| `prompt \| llm_with_tools` | LCEL zinciri — prompt şablonu + araçlı model |

---

## Kullanılan Teknolojiler

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logoColor=white)
![Llama](https://img.shields.io/badge/Llama_3.3_70B-0467DF?style=flat&logo=meta&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)
![Python](https://img.shields.io/badge/Python_3-3776AB?style=flat&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

---

## Kurulum

```bash
pip install langchain-groq langchain-core pydantic
```

### API Anahtarı

Google Colab **Secrets** sekmesine `GROQ_API_KEY` ekle.  
Groq API anahtarı almak için → [console.groq.com](https://console.groq.com)

---

## Kullanım

### Tek fonksiyon

```python
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Fonksiyon şemasını tanımla
class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

# 2. Modele tanıt
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools([WeatherSearch])

# 3. Chain kur ve çağır
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])
chain = prompt | llm_with_tools

response = chain.invoke({"input": "What is the weather in IST today?"})
print(response.tool_calls)
# [{'name': 'WeatherSearch', 'args': {'airport_code': 'IST'}, 'type': 'tool_call'}]
```

### Birden fazla fonksiyon — model doğru olanı seçer

```python
class FlightsSearch(BaseModel):
    """Call this to get flights between two cities"""
    origin: str = Field(description="Departure city code e.g. IST")
    destination: str = Field(description="Arrival city code e.g. LHR")
    date: str = Field(description="Travel date in YYYY-MM-DD format")

# İkisini birden tanıt
llm_with_tools = llm.bind_tools([WeatherSearch, FlightsSearch])
chain = prompt | llm_with_tools

# Hava sorusu → WeatherSearch seçer
response = chain.invoke({"input": "What is the weather in IST today?"})
print(response.tool_calls)
# [{'name': 'WeatherSearch', 'args': {'airport_code': 'IST'}}]

# Uçuş + hava sorusu → ikisini birden çağırır
response = chain.invoke({
    "input": "Ankara'dan Londra'ya 2026-12-15 uçuşları ver, oradaki hava durumu nasıl?"
})
print(response.tool_calls)
# [{'name': 'FlightsSearch', 'args': {'origin': 'ESB', 'destination': 'LHR', 'date': '2026-12-15'}},
#  {'name': 'WeatherSearch', 'args': {'airport_code': 'LHR'}}]
```

---

## Önemli Notlar

**Model fonksiyonu çalıştırmaz** — sadece hangi fonksiyonu hangi parametrelerle çağırman gerektiğini söyler. Çalıştırmak sana kalır.

**`Field(description=...)` kritik** — LLM bu açıklamayı okuyarak parametreyi nasıl dolduracağına karar verir. Açıklama eksik veya yanlışsa model yanlış parametre üretir.

**Model seçimi önemli** — küçük modeller (`llama-3.1-8b`) birden fazla tool'u parse etmekte zorlanabilir. Birden fazla fonksiyon kullanıyorsan `llama-3.3-70b-versatile` tercih et.

---

## Seri İçindeki Yeri

Bu notebook, LangChain ile kurulan RAG serisinin **6. adımıdır.**

```
[1] ✅ Mesaj yapısı ve LLM bağlantısı
[2] ✅ PromptTemplate ile şablonlu prompt
[3] ✅ Çoklu zincir kurma ve zincirleri bağlama
[4] ✅ Metin bölme ve embedding
[5] ✅ ChromaDB vektör store ve benzerlik araması
[6] ✅ Function Calling                              ← bu repo
[7]    Uçtan uca RAG pipeline
```

Her adım bir sonrakine köprü kuruyor.  
Serinin tamamını takip etmek için LinkedIn profilimi ziyaret edebilirsin 👇

---

## Bağlantı

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yasir237)

---

## Lisans

MIT
