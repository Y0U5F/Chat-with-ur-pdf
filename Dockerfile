FROM python:3.9-slim

# تثبيت poppler-utils
RUN apt-get update && apt-get install -y poppler-utils && apt-get clean

# نسخ الملفات
COPY . /app
WORKDIR /app

# تثبيت المكتبات
RUN pip install --no-cache-dir -r requirements.txt

# تشغيل التطبيق
CMD ["streamlit", "run", "app.py"]