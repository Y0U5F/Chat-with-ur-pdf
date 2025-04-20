import streamlit as st
from pdf2image import convert_from_path
import base64
from mistralai import Mistral
from pydantic import BaseModel
from enum import Enum
import pycountry
import json
from pathlib import Path
from mistralai.models import ImageURLChunk, TextChunk
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import aiohttp

# إعداد واجهة Streamlit
st.title("Chat with Your PDF 📄")
st.write("ارفع ملف PDF واسأل أي سؤال عن محتواه، وهنستخرج النصوص ونجاوبك!")

# إعداد Mistral Client
# على Streamlit Cloud، الـ API Key لازم يكون في Secrets
api_key = st.secrets["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# إعداد اللغات عشان ندعم استخراج النصوص بلغات مختلفة
languages = {lang.alpha_2: lang.name for lang in pycountry.languages if hasattr(lang, 'alpha_2')}

class LanguageMeta(Enum.__class__):
    def __new__(metacls, cls, bases, classdict):
        for code, name in languages.items():
            classdict[name.upper().replace(' ', '_')] = name
        return super().__new__(metacls, cls, bases, classdict)

class Language(Enum, metaclass=LanguageMeta):
    pass

# تعريف نموذج البيانات المستخرجة من Mistral OCR
class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: list[Language]
    ocr_contents: dict

# دالة لمعالجة صورة بـ OCR مع Retry
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def structured_ocr_async(image_path: str, session: aiohttp.ClientSession) -> StructuredOCR:
    image_file = Path(image_path)
    assert image_file.is_file(), "الملف مش موجود! اتأكد إن الصورة موجودة."

    # تحويل الصورة لـ base64
    encoded_image = base64.b64encode(image_file.read_bytes()).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

    # معالجة الصورة بـ OCR
    image_response = client.ocr.process(document=ImageURLChunk(image_url=base64_data_url), model="mistral-ocr-latest")
    image_ocr_markdown = image_response.pages[0].markdown

    # تحويل النص لـ JSON
    chat_response = client.chat.parse(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(text=(
                        "ده نص الـ OCR بالماركداون:\n"
                        f"<BEGIN_IMAGE_OCR>\n{image_ocr_markdown}\n<END_IMAGE_OCR>.\n"
                        "حوّل النص ده لـ JSON منظم."
                    ))
                ],
            },
        ],
        response_format=StructuredOCR,
        temperature=0
    )
    return chat_response.choices[0].message.parsed

# دالة لتحويل PDF لصور واستخراج النصوص بشكل غير متزامن
async def process_pdf_async(pdf_file):
    # حفظ الـ PDF مؤقتًا
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    
    # تحويل الـ PDF لصور مع DPI أقل لتقليل الحجم
    # ملاحظة: poppler-utils بيتثبّت عن طريق Dockerfile أو .streamlit/packages.txt
    images = convert_from_path("temp.pdf", dpi=100, first_page=1, last_page=10)  # حد أقصى 10 صفحات
    extracted_texts = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, image in enumerate(images):
            image_path = f"page_{i}.jpg"
            image.save(image_path, "JPEG")
            tasks.append(structured_ocr_async(image_path, session))
        
        # جمع النتايج بشكل غير متزامن
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, StructuredOCR):
                extracted_texts.append(result.ocr_contents)
            else:
                st.warning(f"فشل معالجة صفحة: {result}")
    
    return extracted_texts

# دالة للإجابة على أسئلة
def ask_question(question: str, extracted_texts: list) -> str:
    context = "\n".join([str(text) for text in extracted_texts])
    response = client.chat.complete(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": f"السياق: {context}\n\nالسؤال: {question}\n\nجاوب بناءً على السياق."
            }
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# واجهة Streamlit
uploaded_file = st.file_uploader("ارفع ملف PDF هنا", type="pdf")
if uploaded_file:
    st.write("جاري معالجة الملف... ⏳")
    try:
        extracted_texts = asyncio.run(process_pdf_async(uploaded_file))
        st.session_state["extracted_texts"] = extracted_texts
        st.success("تم استخراج النصوص بنجاح! ✅")
        st.write("النصوص المستخرجة:")
        st.json(extracted_texts)
    except Exception as e:
        st.error(f"حصل خطأ أثناء معالجة الملف: {e}")

question = st.text_input("اسأل سؤال عن محتوى الـ PDF:")
if question and "extracted_texts" in st.session_state:
    try:
        answer = ask_question(question, st.session_state["extracted_texts"])
        st.write("الإجابة:")
        st.write(answer)
    except Exception as e:
        st.error(f"حصل خطأ أثناء الإجابة: {e}")
