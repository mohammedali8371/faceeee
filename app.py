# ============================================================
# app.py — بوت تيليجرام كامل (Face Swap + Inpaint)
# يعمل على Render (الخطة المجانية) — كل شيء في ملف واحد
# ============================================================
import io
import os
import asyncio
import logging
import urllib.request

import cv2
import numpy as np
from PIL import Image

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import uvicorn
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

import insightface
from insightface.app import FaceAnalysis

# ============================================================
# ⚙️ الإعدادات — عدّل قيمتين فقط (التوكن والآيدي)
# ============================================================
# 🔴 ضع توكن البوت من @BotFather هنا
TELEGRAM_BOT_TOKEN = "8545282222:AAHq1O-Ue4JnIad7R_JBP9AXIThzVzUcAq4"

# 🔴 ضع رقم الآيدي الخاص بك (من @userinfobot)
ALLOWED_USER_ID = 7958260008

# مسار موديل تبديل الوجوه داخل المشروع
INSWAPPER_PATH = "models/inswapper_128.onnx"
# (اختياري) رابط لتحميل الموديل تلقائيًا إن لم يكن مرفوعًا
INSWAPPER_URL = ""

# مدة "محاكاة الإعلان" بالثواني
AD_DURATION = 5

# ---- يوفّرها Render تلقائيًا ----
PORT = int(os.getenv("PORT", "10000"))
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
WEBHOOK_PATH = f"/webhook/{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL = f"{RENDER_EXTERNAL_URL}{WEBHOOK_PATH}"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("bot")


# ============================================================
# تحميل الموديل تلقائيًا (إن كان هناك رابط ولم يكن الملف موجودًا)
# ============================================================
def ensure_model():
    if os.path.exists(INSWAPPER_PATH):
        return
    if not INSWAPPER_URL:
        raise FileNotFoundError(
            f"موديل inswapper غير موجود: {INSWAPPER_PATH}\n"
            "ارفع الملف مع المشروع أو ضع رابطًا في INSWAPPER_URL."
        )
    os.makedirs(os.path.dirname(INSWAPPER_PATH), exist_ok=True)
    logger.info(f"تحميل الموديل من {INSWAPPER_URL}...")
    urllib.request.urlretrieve(INSWAPPER_URL, INSWAPPER_PATH)
    logger.info("تم تحميل الموديل.")


# ============================================================
# محرك تبديل الوجوه (InsightFace — onnxruntime فقط، بلا torch)
# ============================================================
class FaceSwapEngine:
    def __init__(self, model_path: str):
        ensure_model()
        providers = ["CPUExecutionProvider"]
        # buffalo_sc أخف بكثير من buffalo_l (100MB بدل 300MB)
        # لو الذاكرة تسمح وتريد دقة أعلى: استبدلها بـ buffalo_l
        self.app = FaceAnalysis(name="buffalo_sc", providers=providers)
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        self.swapper = insightface.model_zoo.get_model(
            model_path, providers=providers
        )

    def swap(self, source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        src_faces = self.app.get(source_bgr)
        tgt_faces = self.app.get(target_bgr)
        if not src_faces:
            raise ValueError("لم يتم كشف وجه في صورة الهوية.")
        if not tgt_faces:
            raise ValueError("لم يتم كشف وجه في الصورة الهدف.")
        src_face = max(
            src_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        result = target_bgr.copy()
        for tgt in tgt_faces:
            s = self.swapper.get(result, tgt, src_face, paste_back=True)
            result = s[0] if isinstance(s, tuple) else s
        return result


# ============================================================
# Inpainting خفيف جدًا بـ OpenCV (بدون نماذج، بدون ذاكرة إضافية)
# ============================================================
def inpaint_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    img_np = np.array(image)
    mask_np = np.array(mask.convert("L"))
    # cv2 يتطلب أن تكون المنطقة المراد تعديلها غير صفرية (أبيض)
    result = cv2.inpaint(
        img_np, mask_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )
    return Image.fromarray(result)


# --- تحميل كسول لتوفير الذاكرة عند البدء ---
_face_engine = None

def get_face_engine() -> FaceSwapEngine:
    global _face_engine
    if _face_engine is None:
        _face_engine = FaceSwapEngine(INSWAPPER_PATH)
    return _face_engine


# ============================================================
# دوال مساعدة
# ============================================================
def is_allowed(update: Update) -> bool:
    user = update.effective_user
    return user is not None and user.id == ALLOWED_USER_ID

async def deny(update: Update):
    await update.effective_message.reply_text("⛔ هذا البوت خاص.")

def buffer_to_bgr(buf: io.BytesIO) -> np.ndarray:
    data = np.frombuffer(buf.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("تعذّر فك ترميز الصورة.")
    return img

def bgr_to_buffer(img: np.ndarray) -> io.BytesIO:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("تعذّر ترميز الصورة.")
    buf = io.BytesIO(enc.tobytes())
    buf.seek(0)
    return buf

def buffer_to_pil(buf: io.BytesIO) -> Image.Image:
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def reset_user_state(ctx: ContextTypes.DEFAULT_TYPE, keep_target: bool = True):
    keys = ["expecting_reference", "expecting_mask", "mask_image"]
    if not keep_target:
        keys.append("target_image")
    for k in keys:
        ctx.user_data.pop(k, None)


# ============================================================
# محاكاة الإعلان
# ============================================================
async def run_ad_simulation(message, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await message.reply_text(f"⏳ محاكاة الإعلان — {AD_DURATION} ثانية…")
    for remaining in range(AD_DURATION - 1, -1, -1):
        await asyncio.sleep(1)
        try:
            await msg.edit_text(f"⏳ محاكاة الإعلان — {remaining} ثانية…")
        except Exception:
            pass
    await msg.edit_text("✅ انتهت المحاكاة. يمكنك تنفيذ العملية الآن.")
    ctx.user_data["ad_passed"] = True


# ============================================================
# أوامر البوت
# ============================================================
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return await deny(update)
    await update.message.reply_text(
        "أرسل صورة (الهدف). بعد انتهاء المحاكاة:\n"
        "• /faceswap — تبديل وجه (أرسل صورة الهوية بعدها).\n"
        "• /inpaint — تعديل منطقة (أرسل ماسكًا أبيض/أسود).\n"
        "• /cancel — إلغاء العملية الحالية."
    )

async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return await deny(update)
    reset_user_state(ctx, keep_target=False)
    ctx.user_data.pop("ad_passed", None)
    await update.message.reply_text("تم الإلغاء. أرسل صورة جديدة.")

async def cmd_faceswap(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return await deny(update)
    if not ctx.user_data.get("ad_passed"):
        return await update.message.reply_text("انتهِ من المحاكاة أولاً.")
    if "target_image" not in ctx.user_data:
        return await update.message.reply_text("أرسل الصورة الهدف أولاً.")
    ctx.user_data["expecting_reference"] = True
    await update.message.reply_text("أرسل الآن صورة الهوية.")

async def cmd_inpaint(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return await deny(update)
    if not ctx.user_data.get("ad_passed"):
        return await update.message.reply_text("انتهِ من المحاكاة أولاً.")
    if "target_image" not in ctx.user_data:
        return await update.message.reply_text("أرسل الصورة الهدف أولاً.")
    ctx.user_data["expecting_mask"] = True
    await update.message.reply_text(
        "أرسل الماسك (أبيض = المنطقة التي ستُعدَّل)."
    )


# ============================================================
# معالجة الصور
# ============================================================
async def handle_photo_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return await deny(update)
    if ctx.user_data.get("expecting_reference"):
        return await handle_reference(update, ctx)
    if ctx.user_data.get("expecting_mask"):
        return await handle_mask(update, ctx)
    return await handle_target(update, ctx)

async def handle_target(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.effective_attachment[-1].get_file()
    buf = io.BytesIO()
    await photo.download_to_memory(buf)
    buf.seek(0)
    ctx.user_data["target_image"] = buf
    await run_ad_simulation(update.message, ctx)

async def handle_reference(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.user_data.get("expecting_reference"):
        return
    ref = await update.message.effective_attachment[-1].get_file()
    ref_buf = io.BytesIO()
    await ref.download_to_memory(ref_buf)
    ref_buf.seek(0)
    try:
        target_bgr = buffer_to_bgr(ctx.user_data["target_image"])
        ref_bgr = buffer_to_bgr(ref_buf)
        engine = get_face_engine()
        result = engine.swap(ref_bgr, target_bgr)
        await update.message.reply_photo(
            photo=bgr_to_buffer(result), caption="✅ اكتمل التبديل."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ فشل: {e}")
    finally:
        ctx.user_data["expecting_reference"] = False

async def handle_mask(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.user_data.get("expecting_mask"):
        return
    mask_file = await update.message.effective_attachment[-1].get_file()
    mask_buf = io.BytesIO()
    await mask_file.download_to_memory(mask_buf)
    mask_buf.seek(0)
    try:
        target_img = buffer_to_pil(ctx.user_data["target_image"])
        mask_img = buffer_to_pil(mask_buf).convert("L")
        mask_img = mask_img.resize(target_img.size, Image.NEAREST)
        result = inpaint_image(target_img, mask_img)
        out = io.BytesIO()
        result.save(out, format="JPEG")
        out.seek(0)
        await update.message.reply_photo(photo=out, caption="✅ اكتمل التعديل.")
    except Exception as e:
        await update.message.reply_text(f"❌ فشل: {e}")
    finally:
        ctx.user_data["expecting_mask"] = False


# ============================================================
# خادم الويب (Webhook + Healthcheck)
# ============================================================
async def healthcheck(request):
    return PlainTextResponse("OK")

def build_starlette_app(ptb_app: Application) -> Starlette:
    async def webhook(request):
        data = await request.json()
        update = Update.de_json(data, ptb_app.bot)
        await ptb_app.process_update(update)
        return PlainTextResponse("ok")

    return Starlette(routes=[
        Route(WEBHOOK_PATH, webhook, methods=["POST"]),
        Route("/healthcheck", healthcheck, methods=["GET"]),
    ])


# ============================================================
# نقطة الدخول
# ============================================================
def main():
    if not RENDER_EXTERNAL_URL:
        logger.error("RENDER_EXTERNAL_URL غير موجود. شغّل المشروع على Render.")
        return

    ptb_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .updater(None)  # Webhook، لا Polling
        .build()
    )

    ptb_app.add_handler(CommandHandler("start", cmd_start))
    ptb_app.add_handler(CommandHandler("cancel", cmd_cancel))
    ptb_app.add_handler(CommandHandler("faceswap", cmd_faceswap))
    ptb_app.add_handler(CommandHandler("inpaint", cmd_inpaint))
    ptb_app.add_handler(
        MessageHandler(filters.PHOTO & filters.ChatType.PRIVATE, handle_photo_router)
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ptb_app.initialize())
    loop.run_until_complete(ptb_app.bot.set_webhook(url=WEBHOOK_URL))
    logger.info(f"تم تسجيل Webhook على: {WEBHOOK_URL}")

    uvicorn.run(
        build_starlette_app(ptb_app),
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
