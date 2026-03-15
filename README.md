markdown
# AI Trainer Tool 🧠⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux-blue)](https://www.linux.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-green)](https://www.python.org)
[![Release](https://img.shields.io/github/v/release/secretaryis/ai-trainer-too)](https://github.com/secretaryis/ai-trainer-too/releases)

**AI Trainer Tool** هي أداة سطح مكتب رسومية (GUI) مفتوحة المصدر تعمل على **لينكس**، تهدف إلى تبسيط عملية تدريب نماذج الذكاء الاصطناعي الصغيرة على المعالج (CPU) للمبتدئين. تتكامل الأداة بسلاسة مع **Ollama** لتشغيل النماذج المُدرّبة محلياً واختبارها تفاعلياً.

---

## ✨ الميزات الرئيسية

- **فحص العتاد الذكي**: كشف تلقائي لإمكانيات الجهاز واقتراح أفضل الإعدادات.
- **إدارة النماذج**: تحميل النماذج بسهولة من Hugging Face مع التحقق من التوافق والترخيص.
- **معالجة البيانات**: دعم إدخال النص المباشر أو رفع ملفات PDF.
- **أوضاع تدريب مرنة**: بسيط (للمبتدئين)، كامل (تحكم كامل)، جزئي (تجميد الطبقات).
- **تصدير بصيغ متعددة**: PyTorch، ONNX، Safetensors، GGUF (لـ Ollama).
- **تكامل مع Ollama**: إنشاء وتشغيل النماذج المحلية بضغطة زر.
- **واجهة ثنائية اللغة**: دعم العربية (RTL) والإنجليزية.
- **سمات قابلة للتخصيص**: فاتحة وداكنة.
- **معالج خطوة بخطوة (Wizard)**: يوجّه المبتدئين خلال سير العمل.

---

## 📥 التثبيت

### 1️⃣ **تحميل الملف التنفيذي (AppImage) – الطريقة الموصى بها**
- قم بتحميل أحدث إصدار من [صفحة الإصدارات](https://github.com/secretaryis/ai-trainer-too/releases) (ابحث عن ملف باسم `AI-Trainer-Tool-*.AppImage`).
- اجعل الملف قابلاً للتنفيذ:
  ```bash
  chmod +x AI-Trainer-Tool-*.AppImage
شغّل الأداة بنقرة مزدوجة أو عبر الطرفية:

bash
./AI-Trainer-Tool-*.AppImage
2️⃣ التثبيت من المصدر (للمطورين)
bash
git clone https://github.com/secretaryis/ai-trainer-too.git
cd ai-trainer-too
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py
3️⃣ باستخدام سكربت التثبيت التلقائي
bash
chmod +x install.sh
./install.sh
هذا السكربت سينشئ بيئة افتراضية، يثبت المتطلبات، ويضيف اختصار سطح المكتب.
