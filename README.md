## 📥 التثبيت (محدث)

### 1️⃣ **الملف التنفيذي (AppImage) – الطريقة الموصى بها**  
- حمّل أحدث إصدار من [صفحة الإصدارات](https://github.com/secretaryis/ai-trainer-too/releases) (ابحث عن ملف باسم `AI-Trainer-Tool-*.AppImage`).  
- اجعل الملف قابلاً للتنفيذ:  
  ```bash
  chmod +x AI-Trainer-Tool-*.AppImage
  ```  
- شغّل الأداة بنقرة مزدوجة أو عبر الطرفية:  
  ```bash
  ./AI-Trainer-Tool-*.AppImage
  ```

### 2️⃣ **سكربت التثبيت التلقائي (install.sh)**  
- حمّل السكربت مباشرة من المستودع:  
  ```bash
  wget https://raw.githubusercontent.com/secretaryis/ai-trainer-too/main/install.sh
  # أو
  curl -O https://raw.githubusercontent.com/secretaryis/ai-trainer-too/main/install.sh
  ```  
- اجعله قابلاً للتنفيذ:  
  ```bash
  chmod +x install.sh
  ```  
- شغّله:  
  ```bash
  ./install.sh
  ```  
  السكربت سيقوم بإنشاء بيئة افتراضية، تثبيت المتطلبات، وإضافة اختصار سطح المكتب.

### 3️⃣ **التثبيت من المصدر (للمطورين)**  
```bash
git clone https://github.com/secretaryis/ai-trainer-too.git
cd ai-trainer-too
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

---

**ملاحظة:** تأكد من توفر المتطلبات الأساسية (Python 3.8+، pip) قبل البدء.
