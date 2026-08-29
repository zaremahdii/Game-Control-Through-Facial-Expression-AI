# Facial Expression AI Server

این سرویس دوربین محلی را باز می‌کند، حرکت سر و حالت چهره را پردازش می‌کند و نتیجه را با WebSocket برای Unity می‌فرستد. Unity هیچ فریم تصویری به این سرویس ارسال نمی‌کند.

## نیازمندی‌ها

- Docker Desktop در حال اجرا
- وب‌کم متصل
- پورت `8000` آزاد

## ساخت و اجرای Docker

در PowerShell این پوشه را باز کنید:

```powershell
cd "E:\facial expression\AI"
docker build -t facial-expression-ai:latest .
docker run --rm --name facial-expression-ai -p 8000:8000 facial-expression-ai:latest
```

برای بررسی آماده‌بودن سرویس:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## اتصال Unity

Unity به آدرس زیر وصل می‌شود:

```text
ws://127.0.0.1:8000/ws
```

سرور پیامی شبیه نمونهٔ زیر می‌فرستد:

```json
{
  "direction": "left",
  "emotion": "neutral"
}
```

مقادیر `direction` شامل `left`، `right` و `neutral` هستند. مقادیر `emotion` شامل `anger`، `happiness`، `neutral`، `sadness` و `surprise` هستند.

## مدل‌ها

هر دو فایل زیر برای شروع سرویس لازم هستند و باید در پوشهٔ `Models` باقی بمانند:

```text
Models/hopenet_robust_alpha1.pkl
Models/vit_best_model.pth
```

## نکتهٔ وب‌کم در Docker Desktop ویندوز

Docker Desktop ویندوز معمولاً وب‌کم میزبان را مستقیماً در اختیار کانتینر Linux قرار نمی‌دهد. اگر WebSocket وصل شد اما سرور خطای بازکردن دوربین داد، وب‌کم باید با USB/IP به Docker VM متصل شود یا سرویس در محیط Linux با دسترسی به دستگاه دوربین اجرا شود.
