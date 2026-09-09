"""
Чат-бот «Алина» для сообщества ВКонтакте.

Бот слушает новые сообщения сообщества через VK Long Poll API,
формирует ответ с помощью ИИ-моделей, доступных через ProxyAPI
(OpenAI-совместимый endpoint https://api.proxyapi.ru/openai/v1),
и отправляет ответ обратно пользователю.

Вся конфигурация задаётся через переменные окружения (см. .env.example).
Реальные ключи не хранятся в коде — пользователь вписывает их в файл .env.

База знаний о Zerocoder (knowledge_base.md) при запуске подставляется
в системный промпт бота и служит единственным источником фактов.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from alina_prompt import SYSTEM_PROMPT

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base.md"


def resolve_ca_bundle() -> Optional[Path]:
    """Возвращает путь к локальному CA bundle, либо None, если не задан."""
    bundle = os.getenv("SSL_CA_BUNDLE")
    if not bundle:
        return None
    path = Path(bundle)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        log.warning("SSL_CA_BUNDLE задан, но файл не найден: %s", path)
        return None
    return path


def apply_ssl_ca_bundle() -> None:
    """Подключает локальный корневой сертификат для HTTPS, если он задан.

    На некоторых машинах HTTPS-трафик перехватывается антивирусом или
    корпоративным прокси с собственным корневым CA, которого нет в
    стандартном хранилище Python. Python в этом случае не может проверить
    сертификат сервера, и бот не подключается к VK/ProxyAPI.

    Два слоя решения:
    1. Для httpx (библиотека openai) выставляется переменная SSL_CERT_FILE —
       ssl.create_default_context() подхватывает её при создании контекста.
    2. Для requests (библиотека vk_api) эта переменная ИГНОРИРУЕТСЯ (в requests
       2.32+ переменные REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE не используются),
       поэтому сертификат передаётся явно через session.verify в __init__.
    """
    global CA_BUNDLE_PATH
    CA_BUNDLE_PATH = resolve_ca_bundle()
    if CA_BUNDLE_PATH is not None:
        log.info("Используется локальный CA bundle: %s", CA_BUNDLE_PATH)
        os.environ["SSL_CERT_FILE"] = str(CA_BUNDLE_PATH)
        os.environ["SSL_CERT_BUNDLE"] = str(CA_BUNDLE_PATH)


CA_BUNDLE_PATH: Optional[Path] = None

# --- Настройки ProxyAPI ---
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
PROXY_API_BASE = os.getenv("PROXY_API_BASE", "https://api.proxyapi.ru/openai/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "300"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))

# --- Настройки ВКонтакте ---
VK_TOKEN = os.getenv("VK_TOKEN")
VK_GROUP_ID = int(os.getenv("VK_GROUP_ID") or 0)
VK_API_VERSION = os.getenv("VK_API_VERSION", "5.199")

MAX_MESSAGE_LENGTH = 4000  # лимит VK на длину сообщения (реальный — 4096)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
# Логи библиотек (httpx/openai) — только предупреждения, чтобы не шуметь.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("alina_bot")

apply_ssl_ca_bundle()

# Клиенты импортируются ПОСЛЕ настройки сертификатов: библиотека httpx
# (использует её openai) читает переменную SSL_CERT_FILE в момент создания
# SSL-контекста.
import requests
from openai import OpenAI
from vk_api import VkApi
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
from vk_api.utils import get_random_id


def load_knowledge_base() -> str:
    """Читает файл базы знаний Zerocoder.

    Файл knowledge_base.md встраивается в системный промпт, поэтому модель
    отвечает на вопросы о курсах только на основе этих данных.
    """
    try:
        content = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
        log.info("База знаний загружена: %s", KNOWLEDGE_BASE_PATH.name)
        return content
    except OSError:
        log.warning("Файл базы знаний не найден: %s", KNOWLEDGE_BASE_PATH)
        return ""


def build_system_prompt() -> str:
    """Собирает итоговый системный промпт: инструкции Алины + база знаний."""
    knowledge = load_knowledge_base()
    if knowledge:
        return SYSTEM_PROMPT + "\n\n## БАЗА ЗНАНИЙ ZEROCODER\n\n" + knowledge
    return SYSTEM_PROMPT


def validate_config() -> None:
    """Проверяет, что все обязательные переменные окружения заданы."""
    missing = []
    if not PROXY_API_KEY:
        missing.append("PROXY_API_KEY")
    if not VK_TOKEN:
        missing.append("VK_TOKEN")
    if not VK_GROUP_ID:
        missing.append("VK_GROUP_ID")
    if missing:
        raise SystemExit(
            "Не заданы обязательные переменные окружения: "
            + ", ".join(missing)
            + ". Скопируйте .env.example в .env и заполните ключи."
        )


def split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Разбивает длинный текст на части для отправки в VK."""
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


class AlinaBot:
    """Бот-консультант Zerocoder для сообщества ВКонтакте."""

    def __init__(self) -> None:
        validate_config()
        self.system_prompt = build_system_prompt()

        self.openai = OpenAI(api_key=PROXY_API_KEY, base_url=PROXY_API_BASE)

        http_session = requests.Session()
        if CA_BUNDLE_PATH is not None:
            # requests 2.32+ не читает REQUESTS_CA_BUNDLE, поэтому локальный
            # корневой сертификат передаётся явно в сессию VK.
            http_session.verify = str(CA_BUNDLE_PATH)
        vk_session = VkApi(token=VK_TOKEN, session=http_session, api_version=VK_API_VERSION)
        self.vk = vk_session.get_api()
        self.longpoll = VkBotLongPoll(vk_session, group_id=VK_GROUP_ID)
        if CA_BUNDLE_PATH is not None:
            # VkBotLongPoll создаёт собственную requests.Session() для опроса
            # lp.vk.ru, поэтому локальный корневой сертификат задаётся и ей.
            self.longpoll.session.verify = str(CA_BUNDLE_PATH)

        # История диалогов: user_id -> список сообщений {"role", "content"}
        self.sessions: dict[int, list[dict]] = {}

    # --- Работа с историей диалога ---
    def get_history(self, user_id: int) -> list[dict]:
        return self.sessions.setdefault(user_id, [])

    def reset_history(self, user_id: int) -> None:
        self.sessions.pop(user_id, None)
        log.info("История сброшена для пользователя %s", user_id)

    def _trim_history(self, history: list[dict]) -> list[dict]:
        return history[-HISTORY_LIMIT:]

    # --- Генерация ответа через ProxyAPI ---
    def generate_answer(self, user_id: int, text: str) -> str:
        history = self.get_history(user_id)
        history.append({"role": "user", "content": text})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._trim_history(history))

        try:
            response = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.7,
            )
            answer = response.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": answer})
            self.sessions[user_id] = self._trim_history(history)
            return answer
        except Exception as exc:  # noqa: BLE001 — ответ пользователю в любом случае
            log.exception("Ошибка запроса к ProxyAPI (модель %s)", OPENAI_MODEL)
            log.debug("Детали: %s", exc)
            return (
                "Не получилось сформировать ответ прямо сейчас. "
                "Попробуйте написать чуть позже, а если вопрос срочный — "
                "оставьте заявку, и наш менеджер свяжется с вами."
            )

    # --- Отправка сообщений в VK ---
    def send_message(self, peer_id: int, text: str) -> None:
        for chunk in split_message(text):
            try:
                self.vk.messages.send(
                    peer_id=peer_id,
                    message=chunk,
                    random_id=get_random_id(),
                    dont_parse_links=1,
                )
            except Exception:  # noqa: BLE001
                log.exception("Не удалось отправить сообщение пользователю %s", peer_id)
            time.sleep(0.4)  # защита от лимитов VK на частоту сообщений

    # --- Обработка команд и сообщений ---
    def handle_message(self, event) -> None:
        message = event.object.get("message", {})
        text = (message.get("text") or "").strip()
        user_id = message.get("from_id")
        peer_id = message.get("peer_id") or user_id

        # Пропускаем служебные события (вступление в беседу, вложения и т. п.)
        if not text or not user_id or message.get("action"):
            return

        if text.startswith("/start"):
            self.reset_history(user_id)
            self.send_message(
                peer_id,
                "Привет! Я Алина, консультант онлайн-университета Zerocoder. "
                "Помогу подобрать направление обучения: нейросети, зерокодинг, "
                "внедрение ИИ в бизнес и не только. Напишите, что вас интересует — "
                "конкретный курс или помощь с выбором направления.",
            )
            return

        if text.startswith("/help"):
            self.send_message(
                peer_id,
                "Я расскажу о направлениях обучения Zerocoder, программах, "
                "форматах и помогу выбрать подходящий курс. Команды: "
                "/start — приветствие, /reset — начать разговор заново, "
                "/help — эта справка. Если нужна персональная консультация — "
                "оставьте заявку, и наш менеджер свяжется с вами.",
            )
            return

        if text.startswith("/reset"):
            self.reset_history(user_id)
            self.send_message(peer_id, "Начинаем разговор заново. О чём хотите узнать?")
            return

        log.info("Сообщение от %s: %s", user_id, text)
        answer = self.generate_answer(user_id, text)
        self.send_message(peer_id, answer)

    # --- Главный цикл Long Poll ---
    def run(self) -> None:
        log.info(
            "Бот запущен. Сообщество: %s, модель: %s, база: %s",
            VK_GROUP_ID,
            OPENAI_MODEL,
            PROXY_API_BASE,
        )
        while True:
            try:
                for event in self.longpoll.listen():
                    if event.type != VkBotEventType.MESSAGE_NEW:
                        log.debug("Событие (пропущено): %s", event.type)
                        continue
                    if not event.from_user:
                        continue
                    self.handle_message(event)
            except KeyboardInterrupt:
                log.info("Бот остановлен.")
                break
            except Exception:  # noqa: BLE001 — переподключение при разрыве
                log.exception("Ошибка в цикле Long Poll, переподключение...")
                time.sleep(3)


def main() -> None:
    AlinaBot().run()


if __name__ == "__main__":
    main()