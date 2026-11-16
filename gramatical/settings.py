"""
Django settings for gramatical project.

Diamankan: SECRET_KEY & DEBUG dibaca dari environment variable (file .env),
tidak lagi hardcoded di dalam kode.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Muat variabel dari file .env (opsional; jika python-dotenv terpasang) ---
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    # python-dotenv belum terpasang → environment variable OS tetap dipakai.
    pass


# SECURITY WARNING: keep the secret key used in production secret!
# Diambil dari env var DJANGO_SECRET_KEY. Fallback HANYA untuk dev lokal.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dev-only-ganti-di-.env-jangan-dipakai-produksi",
)

# SECURITY WARNING: don't run with debug turned on in production!
# DJANGO_DEBUG=False untuk produksi. Default True agar dev lokal lancar.
DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() in ("1", "true", "yes")

# Host yang diizinkan (comma-separated di env var DJANGO_ALLOWED_HOSTS).
ALLOWED_HOSTS = [
    h.strip() for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",") if h.strip()
]
if DEBUG and not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["127.0.0.1", "localhost"]


def _env_bool(nama, default):
    return os.environ.get(nama, str(default)).strip().lower() in ("1", "true", "yes")


# --- Pengaman produksi (aktif otomatis saat DJANGO_DEBUG=False) ---
if not DEBUG:
    from django.core.exceptions import ImproperlyConfigured
    if SECRET_KEY.startswith("django-insecure"):
        raise ImproperlyConfigured("DJANGO_SECRET_KEY wajib diisi di .env saat DJANGO_DEBUG=False.")
    if not ALLOWED_HOSTS:
        raise ImproperlyConfigured("DJANGO_ALLOWED_HOSTS wajib diisi di .env saat DJANGO_DEBUG=False.")
    # DJANGO_HTTPS=False hanya untuk uji sementara tanpa sertifikat (cookie tidak dikunci ke HTTPS).
    _HTTPS = _env_bool("DJANGO_HTTPS", True)
    SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = _HTTPS
    SECURE_SSL_REDIRECT = _HTTPS and _env_bool("DJANGO_SSL_REDIRECT", False)   # biasanya ditangani nginx
    SECURE_HSTS_SECONDS = int(os.environ.get("DJANGO_HSTS_SECONDS", "0"))      # isi mis. 31536000 setelah HTTPS stabil
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_REFERRER_POLICY = "same-origin"
    X_FRAME_OPTIONS = "DENY"

# Di belakang nginx/reverse proxy: percaya header X-Forwarded-Proto & X-Forwarded-For dari proxy.
# JANGAN diaktifkan bila aplikasi diakses langsung dari internet (header bisa dipalsukan).
BEHIND_PROXY = _env_bool("IMLASCAN_BEHIND_PROXY", False)
if BEHIND_PROXY:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Asal yang dipercaya untuk form POST (wajib untuk HTTPS di belakang proxy), mis. https://imlascan.example.com
CSRF_TRUSTED_ORIGINS = [
    o.strip() for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o.strip()
]

# Halaman /admin/ dimatikan secara default di produksi (tidak dipakai ImlaScan).
ENABLE_ADMIN = _env_bool("DJANGO_ENABLE_ADMIN", DEBUG)

# --- Batas pemakaian ImlaScan (melindungi server & kuota API Gemini) ---
IMLASCAN_MAX_UPLOAD_MB = float(os.environ.get("IMLASCAN_MAX_UPLOAD_MB", "10"))
IMLASCAN_MAX_PAGES = int(os.environ.get("IMLASCAN_MAX_PAGES", "5"))             # halaman PDF per unggahan
IMLASCAN_MAX_TEXT_CHARS = int(os.environ.get("IMLASCAN_MAX_TEXT_CHARS", "20000"))
# "jumlah/detik" per alamat IP. Kosongkan ("") untuk mematikan.
IMLASCAN_RATE_LIMIT = os.environ.get("IMLASCAN_RATE_LIMIT", "30/600")          # semua analisis
IMLASCAN_RATE_LIMIT_OCR = os.environ.get("IMLASCAN_RATE_LIMIT_OCR", "10/600")  # unggahan PDF/gambar (OCR)
# Cache memori untuk pembatas laju. Bawaan Django hanya 300 entri (sepertiganya dibuang saat
# penuh), sehingga dengan banyak pengunjung hitungan pembatas bisa hilang.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "imlascan",
        "OPTIONS": {"MAX_ENTRIES": 20000},
    }
}
# Badan permintaan non-berkas (teks tempel + data pembanding OCR) maksimal 2 MB.
DATA_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = int(IMLASCAN_MAX_UPLOAD_MB * 1024 * 1024)


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'home',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'gramatical.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'gramatical.wsgi.application'


# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
