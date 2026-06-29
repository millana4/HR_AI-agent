"""
Планировщик переиндексации. Запускает run_indexers.py по расписанию.

Расписание — список времён (часы:минуты) по таймзоне контейнера
(TZ=Europe/Moscow задан в Dockerfile). Между запусками спит до ближайшего.
Запускается как отдельный сервис в docker-compose, поэтому индексация
работает «из коробки» после docker compose up, без хостового cron.
"""
import asyncio
import subprocess
import sys
from datetime import datetime, timedelta

from app.core.logging import setup_logging, get_logger

# Время запусков по местному времени контейнера (МСК).
RUN_TIMES = [(4, 0), (12, 0)]

logger = get_logger(__name__)


def _seconds_until_next_run() -> float:
    """Сколько секунд до ближайшего времени из RUN_TIMES."""
    now = datetime.now()
    candidates = []
    for hour, minute in RUN_TIMES:
        run_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if run_at <= now:
            run_at += timedelta(days=1)  # уже прошло сегодня — берём завтра
        candidates.append(run_at)
    nearest = min(candidates)
    return (nearest - now).total_seconds()


def _run_indexers() -> None:
    """Запустить индексацию как подпроцесс (тот же скрипт, что и вручную)."""
    logger.info("Планировщик: запуск переиндексации (--faq --documents)")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/run_indexers.py", "--faq", "--documents"],
            capture_output=True,
            text=True,
            timeout=3600,  # час на прогон с большим запасом
        )
        if result.returncode == 0:
            logger.info("Планировщик: переиндексация завершена успешно")
        else:
            logger.error(
                f"Планировщик: переиндексация упала (код {result.returncode}). "
                f"stderr: {result.stderr[-1000:]}"
            )
    except subprocess.TimeoutExpired:
        logger.error("Планировщик: переиндексация превысила таймаут")
    except Exception as exc:
        logger.error(f"Планировщик: ошибка запуска переиндексации: {exc}")


async def main() -> None:
    setup_logging()
    logger.info(f"Планировщик запущен. Расписание (МСК): {RUN_TIMES}")
    while True:
        wait_s = _seconds_until_next_run()
        next_run = datetime.now() + timedelta(seconds=wait_s)
        logger.info(
            f"Планировщик: следующий запуск в {next_run:%Y-%m-%d %H:%M} "
            f"(через {int(wait_s // 60)} мин)"
        )
        await asyncio.sleep(wait_s)
        _run_indexers()
        # Небольшая пауза, чтобы не зациклиться на той же минуте.
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())