import os
import tempfile
import wave
import json
from io import BytesIO

import assemblyai as aai
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from openai import OpenAI
from PIL import Image

# ==========================================================
# 🧩 Design Tokens + Localization + Extended UX Content
# ==========================================================
APP_VERSION = "3.0.0"
APP_BUILD_DATE = "2026-04-17"

EMOTION_ICON_MAP = {
    "angry": "😠",
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "joy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "sadness": "😢",
    "surprise": "😲",
    "contempt": "😒",
    "calm": "😌",
}

EMOTION_COLOR_MAP = {
    "angry": "#ff4d4f",
    "anger": "#ff4d4f",
    "disgust": "#8e44ad",
    "fear": "#f39c12",
    "happy": "#2ecc71",
    "joy": "#2ecc71",
    "neutral": "#95a5a6",
    "sad": "#3498db",
    "sadness": "#3498db",
    "surprise": "#9b59b6",
    "contempt": "#7f8c8d",
    "calm": "#1abc9c",
}

EMOTION_ARABIC_MAP = {
    "angry": "غضب",
    "anger": "غضب",
    "disgust": "اشمئزاز",
    "fear": "خوف",
    "happy": "سعادة",
    "joy": "فرح",
    "neutral": "حياد",
    "sad": "حزن",
    "sadness": "حزن",
    "surprise": "دهشة",
    "contempt": "ازدراء",
    "calm": "هدوء",
}

WELLBEING_LIBRARY_AR = {
    "low_energy": [
        "خذ 10 دقائق مشي خفيف مع تنفس عميق.",
        "اشرب ماء وقلل المنبهات خلال الساعة القادمة.",
        "اكتب 3 أشياء ممتن لها لتحسين المزاج تدريجياً.",
    ],
    "high_stress": [
        "طبّق تقنية 4-7-8 للتنفس لمدة 3 جولات.",
        "قسّم المهمة الكبيرة إلى 3 مهام صغيرة قابلة للتنفيذ.",
        "خفض التنبيهات الرقمية لمدة 30 دقيقة للتركيز الذهني.",
    ],
    "anxiety_signs": [
        "أعد تسمية الأفكار القلقة: (فكرة وليست حقيقة).",
        "ركّز على ما يمكنك التحكم به خلال اليوم فقط.",
        "تواصل مع شخص داعم وشارك ما تمر به باختصار.",
    ],
    "sadness_signs": [
        "تعرض لضوء الشمس 10-15 دقيقة صباحاً.",
        "قم بنشاط بسيط يمنحك إحساساً بالإنجاز.",
        "إذا استمر المزاج المنخفض أياماً طويلة، فكر باستشارة مختص.",
    ],
}

WELLBEING_LIBRARY_EN = {
    "low_energy": [
        "Take a 10-minute light walk with deep breathing.",
        "Drink water and reduce stimulants for the next hour.",
        "Write three gratitude points to gently lift mood.",
    ],
    "high_stress": [
        "Use 4-7-8 breathing for three rounds.",
        "Break one large task into three tiny executable tasks.",
        "Silence digital notifications for 30 minutes.",
    ],
    "anxiety_signs": [
        "Relabel anxious thoughts as thoughts, not facts.",
        "Focus only on what is controllable today.",
        "Reach out to a trusted person and share briefly.",
    ],
    "sadness_signs": [
        "Get morning daylight exposure for 10-15 minutes.",
        "Do one simple action that creates momentum.",
        "If low mood persists, consider professional support.",
    ],
}

FAQ_ITEMS_AR = [
    (
        "هل التطبيق بديل عن التشخيص الطبي؟",
        "لا. هذا التطبيق أداة دعم وتحليل أولي وليست بديلاً عن المختص النفسي أو الطبي.",
    ),
    (
        "كيف تُحسب النبرة الصوتية؟",
        "باستخدام مؤشرات تقريبية مثل شدة الصوت ومعدل عبور الصفر في إشارة WAV.",
    ),
    (
        "ما أفضل مدخلات للحصول على نتيجة أدق؟",
        "صورة واضحة بإضاءة جيدة وصوت هادئ بدون ضوضاء خلفية كبيرة.",
    ),
    (
        "هل اللغة تُكتشف تلقائياً؟",
        "نعم، يتم كشف اللغة تلقائياً من النص المفرغ ثم توليد التشخيص باللغة المناسبة.",
    ),
]

FAQ_ITEMS_EN = [
    (
        "Is this app a medical diagnosis tool?",
        "No. It is a supportive AI analysis assistant, not a replacement for professional care.",
    ),
    (
        "How is vocal tone estimated?",
        "Using lightweight signal indicators like RMS volume and zero-crossing rate on WAV audio.",
    ),
    (
        "What inputs give better results?",
        "Clear face image with good lighting and clean voice recording with minimal noise.",
    ),
    (
        "Is language detected automatically?",
        "Yes. Language is inferred from transcript text and report is generated accordingly.",
    ),
]

ANALYSIS_MODE_HINTS = {
    "متوازن (موصى به)": {
        "temperature": 0.7,
        "max_tokens": 850,
        "ui_message": "توازن بين الدقة والسرعة مع شرح متزن.",
    },
    "تحليل سريع": {
        "temperature": 0.5,
        "max_tokens": 550,
        "ui_message": "نتيجة أسرع بملخص مركز.",
    },
    "تحليل معمّق": {
        "temperature": 0.8,
        "max_tokens": 1200,
        "ui_message": "تفصيل أوسع وتوصيات أعمق.",
    },
}


def _safe_lower(value):
    return str(value).strip().lower() if value is not None else ""


def emotion_to_arabic(label):
    key = _safe_lower(label)
    return EMOTION_ARABIC_MAP.get(key, label)


def emotion_icon(label):
    key = _safe_lower(label)
    return EMOTION_ICON_MAP.get(key, "🧠")


def emotion_color(label):
    key = _safe_lower(label)
    return EMOTION_COLOR_MAP.get(key, "#00c6ff")


def safe_percent(value):
    try:
        val = float(value)
    except Exception:
        val = 0.0
    if val < 0:
        val = 0
    if val > 1:
        val = 1
    return f"{val:.2%}"


def normalize_emotions(emotions):
    cleaned = []
    for row in emotions or []:
        label = str(row.get("label", "")).strip()
        score = float(row.get("score", 0))
        if not label:
            continue
        cleaned.append({"label": label, "score": score})
    cleaned.sort(key=lambda x: x["score"], reverse=True)
    return cleaned


def top_emotions_summary(emotions, top_n=3):
    top = normalize_emotions(emotions)[:top_n]
    chunks = []
    for item in top:
        icon = emotion_icon(item["label"])
        ar = emotion_to_arabic(item["label"])
        chunks.append(f"{icon} {ar} ({safe_percent(item['score'])})")
    return " | ".join(chunks) if chunks else "غير متوفر"


def build_sentiment_digest(sentiments):
    if not sentiments:
        return {"count": 0, "positive": 0, "negative": 0, "neutral": 0}

    digest = {"count": len(sentiments), "positive": 0, "negative": 0, "neutral": 0}
    for item in sentiments:
        val = _safe_lower(getattr(item, "sentiment", ""))
        if "pos" in val:
            digest["positive"] += 1
        elif "neg" in val:
            digest["negative"] += 1
        else:
            digest["neutral"] += 1
    return digest


def sentiment_digest_text(sentiments, lang="ar"):
    d = build_sentiment_digest(sentiments)
    if lang == "en":
        return (
            f"Segments: {d['count']} | Positive: {d['positive']} | "
            f"Negative: {d['negative']} | Neutral: {d['neutral']}"
        )
    return (
        f"عدد المقاطع: {d['count']} | إيجابي: {d['positive']} | "
        f"سلبي: {d['negative']} | محايد: {d['neutral']}"
    )


def pick_recommendations(dominant_label, tone_message, lang="ar"):
    label_key = _safe_lower(dominant_label)
    tone_key = _safe_lower(tone_message)
    if lang == "en":
        lib = WELLBEING_LIBRARY_EN
    else:
        lib = WELLBEING_LIBRARY_AR

    bundle = []
    if "ang" in label_key or "fear" in label_key:
        bundle.extend(lib["high_stress"][:2])
    if "sad" in label_key:
        bundle.extend(lib["sadness_signs"][:2])
    if "quiet" in tone_key or "هادئ" in tone_key:
        bundle.extend(lib["low_energy"][:1])
    if "مرت" in tone_key or "high" in tone_key:
        bundle.extend(lib["anxiety_signs"][:1])
    if not bundle:
        bundle.extend(lib["low_energy"][:2])
    return bundle[:4]


def build_quick_report_payload(
    image_emotion,
    image_score,
    tone_message,
    transcript,
    lang,
    sentiments,
    analysis_mode,
):
    return {
        "app": "Emotion AI Pro — NeuroVision",
        "version": APP_VERSION,
        "build_date": APP_BUILD_DATE,
        "language": lang,
        "analysis_mode": analysis_mode,
        "image_emotion": image_emotion,
        "image_score": image_score,
        "tone": tone_message,
        "transcript_preview": (transcript or "")[:500],
        "sentiment_digest": build_sentiment_digest(sentiments),
    }


def make_markdown_report(payload, diagnosis_text, recommendations):
    lines = [
        "# Emotion AI Pro — NeuroVision Report",
        f"- Version: {payload.get('version')}",
        f"- Build Date: {payload.get('build_date')}",
        f"- Language: {payload.get('language')}",
        f"- Mode: {payload.get('analysis_mode')}",
        "",
        "## Core Signals",
        f"- Image Emotion: {payload.get('image_emotion')} ({safe_percent(payload.get('image_score', 0))})",
        f"- Vocal Tone: {payload.get('tone')}",
        f"- Sentiment Digest: {payload.get('sentiment_digest')}",
        "",
        "## AI Diagnosis",
        diagnosis_text or "No diagnosis generated.",
        "",
        "## Practical Recommendations",
    ]
    for idx, rec in enumerate(recommendations, start=1):
        lines.append(f"{idx}. {rec}")
    lines.append("")
    lines.append("## Transcript Preview")
    lines.append(payload.get("transcript_preview", ""))
    return "\n".join(lines)


def ensure_history_state():
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []


def push_history(record):
    ensure_history_state()
    st.session_state.analysis_history.append(record)
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[-10:]


def render_history_panel():
    ensure_history_state()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕘 آخر التحليلات")
    if not st.session_state.analysis_history:
        st.sidebar.caption("لا يوجد سجل بعد.")
        return
    for idx, item in enumerate(reversed(st.session_state.analysis_history), start=1):
        st.sidebar.caption(
            f"{idx}) {item.get('emotion', 'n/a')} • {item.get('lang', 'n/a')} • {item.get('mode', 'n/a')}"
        )


def render_faq_block(lang="ar"):
    faq_items = FAQ_ITEMS_EN if lang == "en" else FAQ_ITEMS_AR
    with st.expander("❓ الأسئلة الشائعة / FAQ", expanded=False):
        for q, a in faq_items:
            st.markdown(f"**{q}**")
            st.write(a)


def render_footer():
    st.markdown("---")
    st.caption(
        f"NeuroVision v{APP_VERSION} • Build {APP_BUILD_DATE} • Designed for interactive emotional insights."
    )


# ==========================================================
# 🎨 Page Config (must be the first Streamlit command)
# ==========================================================
st.set_page_config(
    page_title="Emotion AI Pro — NeuroVision",
    page_icon="🧠",
    layout="wide",
)

# ==========================================================
# 🔐 API Keys
# ==========================================================
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

missing_keys = []

if not HUGGINGFACE_API_KEY:
    missing_keys.append("HUGGINGFACE_API_KEY")

if not ASSEMBLYAI_API_KEY:
    missing_keys.append("ASSEMBLYAI_API_KEY")

if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")

if missing_keys:
    st.error(f"❌ المفاتيح الناقصة: {', '.join(missing_keys)}")
    st.info("💡 تأكد من إضافتها في Environment Variables في منصة النشر")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================================
# 🌌 UI Design
# ==========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    direction: rtl;
    font-family: 'Tajawal', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 30%, #0b1020, #111a3a, #1b2a52);
    background-size: 200% 200%;
    animation: moveBG 12s ease infinite;
}

@keyframes moveBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.neo-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
    border-radius: 25px;
    padding: 30px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    transition: 0.4s ease;
}

.neo-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}

.stButton>button {
    background: linear-gradient(135deg,#00c6ff,#0072ff);
    color:white;
    border:none;
    padding:14px 35px;
    border-radius:50px;
    font-size:18px;
    font-weight:600;
    transition:0.3s;
}

.stButton>button:hover {
    transform:scale(1.07);
    box-shadow:0 0 25px #00c6ff;
}

h1 {
    text-align:center;
    font-size:40px;
    font-weight:800;
    margin-bottom:30px;
}

.hero-title {
    text-align:center;
    font-size:48px;
    font-weight:800;
    color:#ffffff;
    margin-bottom:8px;
}

.hero-sub {
    text-align:center;
    color:#d6e4ff;
    font-size:18px;
    margin-bottom:22px;
}

.chip-wrap {
    display:flex;
    gap:10px;
    justify-content:center;
    flex-wrap:wrap;
    margin-bottom:20px;
}

.chip {
    background: rgba(0,198,255,0.15);
    color:#dff6ff;
    border:1px solid rgba(0,198,255,0.35);
    padding:8px 14px;
    border-radius:20px;
    font-size:14px;
}

.section-head {
    font-size:22px;
    font-weight:700;
    margin-bottom:14px;
}

.result-grid {
    display:grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-bottom:16px;
}

.kpi {
    background: rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.18);
    border-radius:16px;
    padding:12px;
    text-align:center;
}

.kpi .label {
    color:#cdd8ff;
    font-size:13px;
}

.kpi .value {
    color:#fff;
    font-size:18px;
    font-weight:700;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.08);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
    padding: 8px 14px;
}

.plotly-graph-div {
    border-radius:20px !important;
    overflow:hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# 🧠 HEADER
# ==========================================================
st.markdown("<div class='hero-title'>🧠 Emotion AI Pro — NeuroVision</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>منصة تحليل عاطفي متعددة الوسائط بتجربة تفاعلية عصرية، أنيقة، وسريعة.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class='chip-wrap'>
  <span class='chip'>🎭 تحليل الوجه</span>
  <span class='chip'>🎙️ تحليل الصوت والنبرة</span>
  <span class='chip'>🌍 عربي / English</span>
  <span class='chip'>⚡ تشخيص ذكي فوري</span>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# 📥 INPUT SECTION
# ==========================================================
st.sidebar.markdown("### ⚙️ لوحة التحكم")
analysis_mode = st.sidebar.selectbox(
    "نمط التحليل",
    ["متوازن (موصى به)", "تحليل سريع", "تحليل معمّق"],
    index=0,
)
st.sidebar.caption("يمكنك تغيير النمط لتجربة تفاعلية مختلفة أثناء العرض.")
st.sidebar.info(ANALYSIS_MODE_HINTS[analysis_mode]["ui_message"])
show_faq = st.sidebar.toggle("إظهار FAQ داخل الصفحة", value=True)

tab_image, tab_audio = st.tabs(["📷 الصورة", "🎤 الصوت"])

image_bytes = None
audio_bytes = None
audio_option = "رفع ملف صوتي"

with tab_image:
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'>📷 تحليل تعابير الوجه</div>", unsafe_allow_html=True)

    image_option = st.radio(
        "اختر مصدر الصورة:",
        ["رفع صورة", "التقاط من الكاميرا"],
        horizontal=True,
        key="image_source",
    )
    if image_option == "رفع صورة":
        uploaded_img = st.file_uploader("اختر صورة", type=["jpg", "png"], key="image_uploader")
        if uploaded_img:
            image_bytes = uploaded_img.getvalue()
            st.success("✅ تم رفع الصورة بنجاح.")
    else:
        camera_img = st.camera_input("التقط صورة", key="camera_input")
        if camera_img:
            image_bytes = camera_img.getvalue()
            st.success("✅ تم التقاط الصورة بنجاح.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_audio:
    st.markdown("<div class='neo-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'>🎤 تحليل المشاعر الصوتية</div>", unsafe_allow_html=True)

    audio_option = st.radio(
        "اختر مصدر الصوت:",
        ["رفع ملف صوتي", "تسجيل من الميكروفون"],
        horizontal=True,
        key="audio_source",
    )

    if audio_option == "رفع ملف صوتي":
        uploaded_audio = st.file_uploader(
            "اختر صوت",
            type=["mp3", "wav", "m4a"],
            key="audio_uploader",
        )
        if uploaded_audio:
            audio_bytes = uploaded_audio.getvalue()
            st.audio(audio_bytes)
            st.success("✅ تم رفع الملف الصوتي.")
    else:
        recorded_audio = st.audio_input("سجل صوتك", key="audio_recorder")
        if recorded_audio:
            audio_bytes = recorded_audio.getvalue()
            st.audio(audio_bytes)
            st.success("✅ تم تسجيل الصوت.")
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# 📷 IMAGE ANALYSIS
# ==========================================================
def analyze_image(image_bytes):
    try:
        api_url = "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression"

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/octet-stream",
        }

        response = requests.post(api_url, headers=headers, data=image_bytes, timeout=60)

        response.raise_for_status()

        result = response.json()
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]

        if not isinstance(result, list) or len(result) == 0:
            st.error("❌ لم يتم العثور على وجه.")
            return None, None

        valid_rows = [row for row in result if "label" in row and "score" in row]
        if not valid_rows:
            st.error("❌ نتيجة تحليل الصورة غير صالحة.")
            return None, None

        dominant = max(valid_rows, key=lambda x: x["score"])

        return valid_rows, dominant

    except Exception as e:
        st.error(f"❌ خطأ: {e}")
        return None, None


# ==========================================================
# 🎤 AUDIO ANALYSIS
# ==========================================================
def analyze_audio(audio_bytes):
    tmp_path = None
    try:
        aai.settings.api_key = ASSEMBLYAI_API_KEY

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        config = aai.TranscriptionConfig(sentiment_analysis=True)

        transcript = aai.Transcriber().transcribe(tmp_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"❌ فشل التفريغ: {transcript.error}")
            return None, None

        return transcript.text or "لا يوجد نص.", transcript.sentiment_analysis or []

    except Exception as e:
        st.error(f"❌ خطأ في الصوت: {e}")
        return None, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==========================================================
# 🎚️ TONE ANALYSIS (Pitch + Volume) for Recorded WAV
# ==========================================================
def _zero_crossing_rate(samples):
    if len(samples) < 2:
        return 0.0

    crossings = 0
    prev = samples[0]
    for current in samples[1:]:
        if (prev < 0 <= current) or (prev >= 0 > current):
            crossings += 1
        prev = current
    return crossings / (len(samples) - 1)


def _normalize_audio_samples(raw_frames, sample_width):
    if sample_width == 1:
        return [int(b) - 128 for b in raw_frames]
    if sample_width == 2:
        import array

        return array.array("h", raw_frames).tolist()
    if sample_width == 4:
        import array

        return array.array("i", raw_frames).tolist()
    return []


def analyze_tone(audio_bytes):
    """
    Lightweight tone estimation for WAV audio:
    - Volume (RMS amplitude)
    - Zero crossing rate (rough pitch activity indicator)
    """
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)

        samples = _normalize_audio_samples(raw_frames, sample_width)
        if not samples:
            return {
                "status": "unsupported_format",
                "message": "تعذر تحليل النبرة (تنسيق غير مدعوم).",
            }

        # Downsample for speed on large recordings
        step = max(1, int(len(samples) / 50000))
        reduced = samples[::step]

        max_abs = max(abs(x) for x in reduced) or 1
        rms = (sum(x * x for x in reduced) / len(reduced)) ** 0.5
        rms_ratio = rms / max_abs
        zcr = _zero_crossing_rate(reduced)

        if zcr > 0.18:
            pitch_label = "نبرة مرتفعة/سريعة"
        elif zcr > 0.1:
            pitch_label = "نبرة متوسطة"
        else:
            pitch_label = "نبرة منخفضة/هادئة"

        if rms_ratio > 0.45:
            volume_label = "صوت قوي"
        elif rms_ratio > 0.25:
            volume_label = "صوت متوسط"
        else:
            volume_label = "صوت هادئ"

        return {
            "status": "ok",
            "frame_rate": frame_rate,
            "zcr": round(zcr, 4),
            "rms_ratio": round(rms_ratio, 4),
            "pitch_label": pitch_label,
            "volume_label": volume_label,
            "message": f"{pitch_label} — {volume_label}",
        }
    except wave.Error:
        return {
            "status": "unsupported_format",
            "message": "تحليل النبرة متاح حالياً فقط للصوت المسجل بصيغة WAV.",
        }
    except Exception as e:
        return {"status": "error", "message": f"فشل تحليل النبرة: {e}"}


# ==========================================================
# 🌍 AUTO LANGUAGE DIAGNOSIS
# ==========================================================
def generate_diagnosis(image_emotion, audio_text, tone_summary, analysis_mode):
    try:
        try:
            detected_lang = detect(audio_text) if audio_text.strip() else "ar"
        except Exception:
            detected_lang = "ar"

        if detected_lang == "en":
            system_msg = """
You are an expert emotional intelligence and psychology AI analyst.
Provide a professional emotional evaluation based on facial emotion and speech.
Your analysis must be structured, supportive and informative.
"""

            final_prompt = f"""
Facial Emotion Detected:
{image_emotion}

Speech Transcript:
{audio_text}

Tone Analysis:
{tone_summary}

Analysis Mode:
{analysis_mode}

Provide a professional emotional analysis with:

1. Emotional interpretation
2. Psychological indicators:
   Anxiety
   Stress
   Depression
3. Emotional state summary
4. Practical wellbeing recommendations
"""

        else:
            system_msg = """
أنت خبير تحليل نفسي وعاطفي باستخدام الذكاء الاصطناعي.
قدم تحليلًا احترافيًا بناءً على تعبير الوجه والنص الصوتي.
"""

            final_prompt = f"""
العاطفة من الصورة:
{image_emotion}

النص من الصوت:
{audio_text}

تحليل النبرة:
{tone_summary}

نمط التحليل:
{analysis_mode}

قدم تحليل يتضمن:

1 تفسير الحالة العاطفية
2 تقدير احتمالية:
القلق
التوتر
الاكتئاب
3 ملخص الحالة
4 توصيات عملية لتحسين الحالة النفسية
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": final_prompt},
            ],
            temperature=ANALYSIS_MODE_HINTS[analysis_mode]["temperature"],
            max_tokens=ANALYSIS_MODE_HINTS[analysis_mode]["max_tokens"],
        )

        content = response.choices[0].message.content if response.choices else None
        return content or "تعذر إنشاء التشخيص حالياً.", detected_lang

    except Exception as e:
        st.error(f"❌ خطأ في OpenAI: {e}")
        return None, None


# ==========================================================
# 🚀 START ANALYSIS
# ==========================================================
if st.button("🚀 بدء التحليل الذكي"):
    if image_bytes and audio_bytes:
        st.toast(f"🚀 تم بدء التحليل — النمط: {analysis_mode}")
        progress = st.progress(0, text="تهيئة التحليل...")

        progress.progress(20, text="تحليل تعابير الوجه...")

        emotions, dominant = analyze_image(image_bytes)

        if not emotions:
            st.stop()

        progress.progress(50, text="تفريغ وتحليل الصوت...")

        text, sentiments = analyze_audio(audio_bytes)

        if text is None:
            st.stop()

        progress.progress(70, text="تحليل النبرة ودمج النتائج...")

        tone_result = None
        if audio_option == "تسجيل من الميكروفون":
            tone_result = analyze_tone(audio_bytes)

        tone_summary = (
            tone_result["message"]
            if tone_result and tone_result.get("message")
            else "غير متوفر"
        )

        diagnosis, lang = generate_diagnosis(
            dominant["label"], text, tone_summary, analysis_mode
        )

        progress.progress(100, text="اكتمل التحليل ✅")

        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.image(Image.open(BytesIO(image_bytes)), use_container_width=True)

        with col_b:
            labels = [e["label"] for e in emotions]
            scores = [e["score"] for e in emotions]
            bar_colors = [emotion_color(lbl) for lbl in labels]

            fig = go.Figure([go.Bar(x=labels, y=scores, marker_color=bar_colors)])

            fig.update_layout(title="تحليل تعابير الوجه", template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='neo-card'>", unsafe_allow_html=True)

        st.subheader("📊 التشخيص النهائي")

        dominant_score = next(
            (e["score"] for e in emotions if e["label"] == dominant["label"]),
            0,
        )
        st.markdown(
            f"""
<div class='result-grid'>
  <div class='kpi'><div class='label'>العاطفة المسيطرة</div><div class='value'>{emotion_icon(dominant['label'])} {emotion_to_arabic(dominant['label'])}</div></div>
  <div class='kpi'><div class='label'>درجة الثقة</div><div class='value'>{dominant_score:.2%}</div></div>
  <div class='kpi'><div class='label'>نمط التحليل</div><div class='value'>{analysis_mode}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.info(f"🌍 اللغة المكتشفة: {'العربية' if lang == 'ar' else 'English'}")
        if tone_result:
            st.success(f"🎚️ تحليل النبرة: {tone_result['message']}")
        if sentiments:
            st.caption(f"🧾 عدد مقاطع تحليل المشاعر الصوتية: {len(sentiments)}")
            st.caption(sentiment_digest_text(sentiments, lang))

        st.info(f"🧭 ملخص سريع: {top_emotions_summary(emotions, top_n=3)}")

        st.write(diagnosis)

        recommendations = pick_recommendations(dominant["label"], tone_summary, lang)
        st.markdown("#### ✅ توصيات عملية سريعة")
        for rec in recommendations:
            st.markdown(f"- {rec}")

        payload = build_quick_report_payload(
            image_emotion=dominant["label"],
            image_score=dominant_score,
            tone_message=tone_summary,
            transcript=text,
            lang=lang,
            sentiments=sentiments,
            analysis_mode=analysis_mode,
        )

        markdown_report = make_markdown_report(payload, diagnosis, recommendations)
        col_download_1, col_download_2 = st.columns(2)
        with col_download_1:
            st.download_button(
                "⬇️ تحميل التقرير (Markdown)",
                data=markdown_report,
                file_name="emotion_report.md",
                mime="text/markdown",
            )
        with col_download_2:
            st.download_button(
                "⬇️ تحميل البيانات (JSON)",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name="emotion_report.json",
                mime="application/json",
            )

        push_history(
            {
                "emotion": dominant["label"],
                "lang": lang,
                "mode": analysis_mode,
            }
        )

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠ يرجى إدخال صورة وصوت أولاً.")

if show_faq:
    render_faq_block("ar")

render_history_panel()


# === EXTENDED_LIBRARY_BLOCK_START ===
EXTENDED_REFLECTION_PROMPTS = [
    "صف شعورك الحالي بكلمة واحدة.",
    "ما الحدث الذي أثر على مزاجك اليوم؟",
    "ما الشيء الصغير الذي قد يحسن يومك الآن؟",
    "كيف يمكن أن تدعم نفسك خلال الساعة القادمة؟",
    "ما الفكرة التي تحتاج إعادة صياغة بشكل ألطف؟",
]

EXTENDED_MICRO_ACTIONS = [
    "تنفس بعمق 4-6 مرات ببطء.",
    "اشرب كوب ماء وخذ استراحة دقيقتين.",
    "قم بتمدد بسيط للرقبة والكتفين.",
    "اكتب 3 أولويات قصيرة لباقي اليوم.",
    "ابتعد عن الشاشة 5 دقائق.",
]

EXTENDED_AFFIRMATIONS_AR = [
    "أنا أتعامل مع يومي بخطوات صغيرة وواثقة.",
    "أنا أستحق الراحة وإعادة التوازن.",
    "كل خطوة صغيرة تصنع فرقًا.",
    "أستطيع طلب الدعم عند الحاجة.",
    "مشاعري مهمة ويمكنني فهمها بهدوء.",
]


def get_extended_prompts(limit=5):
    safe_limit = max(1, min(int(limit), len(EXTENDED_REFLECTION_PROMPTS)))
    return EXTENDED_REFLECTION_PROMPTS[:safe_limit]


def get_extended_actions(limit=5):
    safe_limit = max(1, min(int(limit), len(EXTENDED_MICRO_ACTIONS)))
    return EXTENDED_MICRO_ACTIONS[:safe_limit]


def get_extended_affirmations(limit=5):
    safe_limit = max(1, min(int(limit), len(EXTENDED_AFFIRMATIONS_AR)))
    return EXTENDED_AFFIRMATIONS_AR[:safe_limit]


def render_extended_coaching_block():
    with st.expander("🧠 مكتبة دعم مختصرة", expanded=False):
        st.markdown("#### أسئلة انعكاسية")
        for item in get_extended_prompts(5):
            st.markdown(f"- {item}")

        st.markdown("#### إجراءات سريعة")
        for item in get_extended_actions(5):
            st.markdown(f"- {item}")

        st.markdown("#### عبارات دعم ذاتي")
        for item in get_extended_affirmations(5):
            st.markdown(f"- {item}")


render_extended_coaching_block()
render_footer()
# === EXTENDED_LIBRARY_BLOCK_END ===
