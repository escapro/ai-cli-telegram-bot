import asyncio
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from os import getenv
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv

load_dotenv()

TOKEN = getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_IDS = {int(uid.strip()) for uid in getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip()}
CLAUDE_PATH = getenv("CLAUDE_PATH", "/home/ruslan/.local/bin/claude")
WORK_DIR = getenv("WORK_DIR", "/home/ruslan")
CLAUDE_TIMEOUT = int(getenv("CLAUDE_TIMEOUT", "300"))
SESSIONS_FILE = Path(__file__).parent / "sessions.json"
MAX_MESSAGE_LENGTH = 4096
# stream-json отдаёт одно событие на строку; строка (крупный Read, вывод Bash)
# легко превышает дефолтный лимит StreamReader в 64 КБ, из-за чего readline()
# падает с "Separator is not found, and chunk exceed the limit".
STREAM_LIMIT = 64 * 1024 * 1024  # 64 МБ на строку
AVAILABLE_MODELS = ["opus", "sonnet", "haiku"]
DEFAULT_MODEL = "sonnet"

router = Router()

# --- Sessions ---

def load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        data = json.loads(SESSIONS_FILE.read_text())
        data.setdefault("model", DEFAULT_MODEL)
        data.setdefault("thinking", False)
        data.setdefault("plain", False)
        data.setdefault("fast", False)
        return data
    return {"current": None, "sessions": {}, "model": DEFAULT_MODEL, "thinking": False, "plain": False, "fast": False}


def save_sessions(data: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# --- Claude CLI ---

TOOL_LABELS = {
    "Read": ("📖", "Читаю"),
    "Edit": ("✏️", "Редактирую"),
    "Write": ("📝", "Создаю"),
    "Bash": ("💻", "Команда"),
    "Grep": ("🔍", "Ищу в коде"),
    "Glob": ("📂", "Ищу файлы"),
    "Agent": ("🤖", "Агент"),
    "WebSearch": ("🌐", "Поиск в сети"),
    "WebFetch": ("🌐", "Загружаю страницу"),
}
MIN_EDIT_INTERVAL = 1.0


def format_tool_line(name: str, input_data: dict) -> str:
    emoji, label = TOOL_LABELS.get(name, ("🔧", name))
    detail = ""
    if name in ("Read", "Edit", "Write") and input_data.get("file_path"):
        detail = Path(input_data["file_path"]).name
    elif name == "Bash" and input_data.get("command"):
        detail = input_data["command"][:50]
    elif name == "Grep" and input_data.get("pattern"):
        detail = input_data["pattern"][:30]
    elif name == "Glob" and input_data.get("pattern"):
        detail = input_data["pattern"][:30]
    elif name == "Agent" and input_data.get("description"):
        detail = input_data["description"][:40]
    return f"{emoji} {label} {detail}".strip()


async def call_claude_streaming(
    prompt: str,
    session_id: str | None = None,
    model: str = DEFAULT_MODEL,
    thinking: bool = False,
    plain: bool = False,
    fast: bool = False,
    on_tool=None,
) -> dict:
    cmd = [
        CLAUDE_PATH, "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--model", model,
        "--add-dir", "/home/ruslan",
        "--add-dir", "/home/ruslan/telegram-bot",
    ]
    if thinking:
        cmd.extend(["--effort", "max"])
    elif fast:
        cmd.extend(["--effort", "low"])
    if plain:
        cmd.extend(["--tools", ""])
    if session_id:
        cmd.extend(["--resume", session_id])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=WORK_DIR,
        limit=STREAM_LIMIT,
    )
    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    result_data = {}

    async for raw_line in proc.stdout:
        line = raw_line.decode().strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")

        if event_type == "assistant" and on_tool:
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_line = format_tool_line(block["name"], block.get("input", {}))
                    await on_tool(tool_line)

        elif event_type == "result":
            result_data = {
                "session_id": event.get("session_id"),
                "result": event.get("result", ""),
            }
            if event.get("is_error"):
                result_data["error"] = event.get("result", "Unknown error")

    await proc.wait()

    if proc.returncode != 0 and "error" not in result_data:
        result_data["error"] = f"Claude CLI exited with code {proc.returncode}"

    return result_data


# --- Helpers ---

async def send_long_message(message: Message, text: str, parse_mode: str | None = None) -> None:
    if not text:
        await message.answer("(пустой ответ)")
        return

    for i in range(0, len(text), MAX_MESSAGE_LENGTH):
        chunk = text[i:i + MAX_MESSAGE_LENGTH]
        try:
            await message.answer(chunk, parse_mode=parse_mode)
        except Exception:
            await message.answer(chunk, parse_mode=None)


# --- Auth middleware ---

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


# --- Command handlers ---

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    await message.answer(
        "Claude CLI Telegram Bot\n\n"
        "Отправь сообщение — я передам его в Claude и верну ответ.\n\n"
        "Команды:\n"
        "/new — новая сессия\n"
        "/sessions — список сессий\n"
        "/switch <id> — переключить сессию\n"
        "/current — текущая сессия\n"
        "/help — помощь"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    await message.answer(
        "Команды:\n"
        "/new — начать новую сессию (сбросить контекст)\n"
        "/sessions — список всех сессий\n"
        "/switch <номер> — переключиться на сессию из списка\n"
        "/current — показать текущую сессию\n"
        "/help — эта справка\n\n"
        "Любое текстовое сообщение будет отправлено в Claude CLI."
    )


@router.message(Command("new"))
async def cmd_new(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    data = load_sessions()
    data["current"] = None
    save_sessions(data)
    await message.answer("Новая сессия. Следующее сообщение начнёт новый диалог.")


@router.message(Command("sessions"))
async def cmd_sessions(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    data = load_sessions()
    sessions = data.get("sessions", {})
    if not sessions:
        await message.answer("Нет сохранённых сессий.")
        return

    lines = []
    for i, (sid, info) in enumerate(sessions.items(), 1):
        marker = " <<" if sid == data.get("current") else ""
        name = info.get("name", "без названия")
        created = info.get("created", "?")
        lines.append(f"{i}. {name}\n   {created}{marker}")

    await send_long_message(message, "Сессии:\n\n" + "\n\n".join(lines))


@router.message(Command("switch"))
async def cmd_switch(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    data = load_sessions()
    sessions = data.get("sessions", {})
    if not sessions:
        await message.answer("Нет сохранённых сессий.")
        return

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer("Использование: /switch <номер>\nНомер из /sessions")
        return

    try:
        idx = int(args[1]) - 1
        session_ids = list(sessions.keys())
        if 0 <= idx < len(session_ids):
            sid = session_ids[idx]
            data["current"] = sid
            save_sessions(data)
            name = sessions[sid].get("name", "без названия")
            await message.answer(f"Переключено на сессию: {name}")
        else:
            await message.answer(f"Неверный номер. Доступно: 1-{len(session_ids)}")
    except ValueError:
        await message.answer("Укажи номер сессии (число).")


def toggle_label(name: str, enabled: bool) -> str:
    return f"{'✅' if enabled else '⬜'} {name}"


def model_keyboard(model: str, thinking: bool, plain: bool, fast: bool) -> InlineKeyboardMarkup:
    model_buttons = []
    for m in AVAILABLE_MODELS:
        label = f"{'> ' if m == model else ''}{m}"
        model_buttons.append(InlineKeyboardButton(text=label, callback_data=f"model:{m}"))
    toggle_buttons = [
        InlineKeyboardButton(text=toggle_label("thinking", thinking), callback_data="toggle:thinking"),
        InlineKeyboardButton(text=toggle_label("plain", plain), callback_data="toggle:plain"),
        InlineKeyboardButton(text=toggle_label("fast", fast), callback_data="toggle:fast"),
    ]
    return InlineKeyboardMarkup(inline_keyboard=[model_buttons, toggle_buttons])


def model_status_text(model: str, thinking: bool, plain: bool, fast: bool) -> str:
    flags = []
    if thinking:
        flags.append("thinking")
    if plain:
        flags.append("plain")
    if fast:
        flags.append("fast")
    suffix = f" | {', '.join(flags)}" if flags else ""
    return f"Модель: {model}{suffix}"


@router.message(Command("model"))
async def cmd_model(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    data = load_sessions()
    model = data.get("model", DEFAULT_MODEL)
    thinking = data.get("thinking", False)
    plain = data.get("plain", False)
    fast = data.get("fast", False)
    await message.answer(
        model_status_text(model, thinking, plain, fast),
        reply_markup=model_keyboard(model, thinking, plain, fast),
    )


@router.callback_query(lambda c: c.data and c.data.startswith("model:"))
async def on_model_selected(callback: CallbackQuery) -> None:
    if not is_allowed(callback.from_user.id):
        await callback.answer("Нет доступа")
        return
    model = callback.data.split(":", 1)[1]
    if model not in AVAILABLE_MODELS:
        await callback.answer("Неизвестная модель")
        return
    data = load_sessions()
    data["model"] = model
    save_sessions(data)
    thinking = data.get("thinking", False)
    plain = data.get("plain", False)
    fast = data.get("fast", False)
    await callback.message.edit_text(
        model_status_text(model, thinking, plain, fast),
        reply_markup=model_keyboard(model, thinking, plain, fast),
    )
    await callback.answer(f"Модель: {model}")


@router.callback_query(lambda c: c.data and c.data.startswith("toggle:"))
async def on_toggle(callback: CallbackQuery) -> None:
    if not is_allowed(callback.from_user.id):
        await callback.answer("Нет доступа")
        return
    key = callback.data.split(":", 1)[1]
    if key not in ("thinking", "plain", "fast"):
        await callback.answer("Неизвестный параметр")
        return
    data = load_sessions()
    data[key] = not data.get(key, False)
    # thinking и fast взаимоисключающие
    if key == "thinking" and data[key]:
        data["fast"] = False
    elif key == "fast" and data[key]:
        data["thinking"] = False
    save_sessions(data)
    model = data.get("model", DEFAULT_MODEL)
    thinking = data.get("thinking", False)
    plain = data.get("plain", False)
    fast = data.get("fast", False)
    status = "вкл" if data[key] else "выкл"
    await callback.message.edit_text(
        model_status_text(model, thinking, plain, fast),
        reply_markup=model_keyboard(model, thinking, plain, fast),
    )
    await callback.answer(f"{key}: {status}")


@router.message(Command("current"))
async def cmd_current(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return
    data = load_sessions()
    current = data.get("current")
    if not current:
        await message.answer("Нет активной сессии. Следующее сообщение начнёт новую.")
        return
    info = data.get("sessions", {}).get(current, {})
    name = info.get("name", "без названия")
    created = info.get("created", "?")
    await message.answer(f"Текущая сессия: {name}\nСоздана: {created}\nID: {current}")


# --- Main message handler ---

@router.message()
async def handle_message(message: Message) -> None:
    if not is_allowed(message.from_user.id):
        return

    if not message.text:
        await message.answer("Поддерживаются только текстовые сообщения.")
        return

    text = message.text

    data = load_sessions()
    session_id = data.get("current")
    model = data.get("model", DEFAULT_MODEL)
    thinking = data.get("thinking", False)
    plain = data.get("plain", False)
    fast = data.get("fast", False)

    # Status message with live updates
    status_msg = await message.answer("⏳ Думаю...")
    status_lines = []
    last_edit_time = 0.0

    async def on_tool(tool_line: str):
        nonlocal last_edit_time
        status_lines.append(tool_line)
        now = asyncio.get_event_loop().time()
        if now - last_edit_time < MIN_EDIT_INTERVAL:
            return
        last_edit_time = now
        display = "\n".join(status_lines[-5:])
        try:
            await status_msg.edit_text(f"⏳ Работаю...\n\n{display}")
        except Exception:
            pass

    try:
        result = await asyncio.wait_for(
            call_claude_streaming(text, session_id, model, thinking, plain, fast, on_tool),
            timeout=CLAUDE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        try:
            await status_msg.delete()
        except Exception:
            pass
        await message.answer(f"Таймаут: Claude не ответил за {CLAUDE_TIMEOUT} секунд.")
        return
    except Exception as e:
        try:
            await status_msg.delete()
        except Exception:
            pass
        await message.answer(f"Ошибка: {e}")
        return

    # Delete status message
    try:
        await status_msg.delete()
    except Exception:
        pass

    if "error" in result:
        await message.answer(f"Ошибка:\n{result['error']}")
        return

    # Save session
    new_session_id = result.get("session_id")
    if new_session_id:
        tz = timezone(timedelta(hours=4))
        now = datetime.now(tz).isoformat(timespec="seconds")
        if new_session_id not in data.get("sessions", {}):
            if "sessions" not in data:
                data["sessions"] = {}
            name = text[:50] + ("..." if len(text) > 50 else "")
            data["sessions"][new_session_id] = {"name": name, "created": now}
        data["current"] = new_session_id
        save_sessions(data)

    # Send response
    response_text = result.get("result", "(пустой ответ)")
    await send_long_message(message, response_text)


# --- Main ---

async def main() -> None:
    if not TOKEN:
        print("Ошибка: TELEGRAM_BOT_TOKEN не задан в .env")
        sys.exit(1)

    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher()
    dp.include_router(router)

    await bot.set_my_commands([
        BotCommand(command="new", description="Новая сессия"),
        BotCommand(command="model", description="Переключить модель"),
        BotCommand(command="sessions", description="Список сессий"),
        BotCommand(command="switch", description="Переключить сессию"),
        BotCommand(command="current", description="Текущая сессия"),
        BotCommand(command="help", description="Помощь"),
    ])

    logging.info("Бот запущен. Разрешённые пользователи: %s", ALLOWED_USER_IDS or "все")
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
