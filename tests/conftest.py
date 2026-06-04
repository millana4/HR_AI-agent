"""
Глобальная настройка для тестов.

Очищает переменные прокси перед запуском, чтобы httpx-клиенты
внутри qdrant-client и других библиотек не падали на socks://.
"""
import os


_PROXY_VARS = [
    "ALL_PROXY", "all_proxy",
    "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy",
    "FTP_PROXY", "ftp_proxy",
    "NO_PROXY", "no_proxy",
]

for var in _PROXY_VARS:
    os.environ.pop(var, None)