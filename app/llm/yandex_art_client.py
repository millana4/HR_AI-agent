"""
Клиент для Alice AI ART — синхронной генерации изображений.

Модель aliceai-image-art-3.0 работает через OpenAI-совместимый Images API
(sync, без поллинга операций): один POST на /v1/images/generations →
картинка сразу в data[0].b64_json (Base64 JPEG).

ВАЖНО (ПД): Yandex логирует промпты генерации изображений на своей стороне,
отключить заголовком нельзя. Поэтому передаём только МАСКИРОВАННЫЙ промпт
(имена уже заменены на NAME_N) — реальные ПД в модель не уходят.

Аутентификация: Api-Key сервисного аккаунта + роль ai.imageGeneration.user.
"""
import httpx

from app.core.config import Config
from app.core.exceptions import LLMAuthError, LLMError, LLMTimeoutError
from app.core.logging import get_logger


logger = get_logger(__name__)


class YandexArtClient:
    """Клиент генерации изображений Alice AI ART (sync OpenAI Images API)."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=Config.YANDEX_ART_TIMEOUT,
            trust_env=False,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def generate_image(
        self,
        prompt: str,
        correlation_id: str = "-",
    ) -> str:
        """
        Сгенерировать изображение по текстовому промпту (синхронно).

        Args:
            prompt: текст запроса (УЖЕ маскированный — без реальных ПД).

        Returns:
            Base64-строка JPEG-изображения.

        Raises:
            LLMAuthError: 401/403 (ключ/роль/оплата).
            LLMTimeoutError: модель не ответила за отведённое время.
            LLMError: прочие ошибки API.
        """
        model_uri = f"art://{Config.YANDEX_FOLDER_ID}/{Config.YANDEX_ART_MODEL}"
        headers = {
            "Authorization": f"Api-Key {Config.YANDEX_API_KEY}",
            "Content-Type": "application/json",
            "x-folder-id": Config.YANDEX_FOLDER_ID,
        }
        body = {
            "model": model_uri,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        url = f"{Config.YANDEX_ART_API_URL.rstrip('/')}/images/generations"

        logger.info(
            f"YandexART generate: prompt_len={len(prompt)}",
            extra={"correlation_id": correlation_id},
        )

        try:
            resp = await self._http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"YandexART timeout ({Config.YANDEX_ART_TIMEOUT}s)"
            ) from exc
        except Exception as exc:
            raise LLMError(f"YandexART request failed: {exc}") from exc

        if resp.status_code in (401, 403):
            raise LLMAuthError(
                f"YandexART auth error: HTTP {resp.status_code}, "
                f"body: {resp.text[:300]}"
            )
        if resp.status_code != 200:
            raise LLMError(
                f"YandexART error: HTTP {resp.status_code}, "
                f"body: {resp.text[:300]}"
            )

        data = resp.json()
        items = data.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise LLMError(
                f"YandexART: no image in response: {str(data)[:300]}"
            )

        image_b64 = items[0]["b64_json"]
        logger.info(
            f"YandexART image ready: {len(image_b64)} b64 chars",
            extra={"correlation_id": correlation_id},
        )
        return image_b64