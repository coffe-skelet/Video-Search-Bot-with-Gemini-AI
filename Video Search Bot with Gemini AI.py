import os
import csv
import json
import time
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import google.generativeai as genai
import telebot
from telebot import types

# ---------------------- НАСТРОЙКИ (ЗАПОЛНИТЕ ПЕРЕД ЗАПУСКОМ) ----------------------
# API ключи
GEMINI_API_KEY = "ВАШ_API_КЛЮЧ_GEMINI"        # 🔑 API-ключ Gemini
TELEGRAM_BOT_TOKEN = "ВАШ_ТОКЕН_БОТА"         # 🤖 Токен Telegram бота от @BotFather

# Пути
FOLDER_PATH = "/путь/к/папке/с/видео"         # 📁 Путь к папке с видеофайлами
OUTPUT_CSV = "video_themes.csv"               # 📄 Имя выходного CSV-файла
EMBEDDINGS_FILE = "embeddings.pkl"            # 💾 Файл для сохранения эмбеддингов

# Настройки Gemini
BATCH_SIZE = 150                              # Количество названий в одном запросе для категоризации
EMBEDDING_BATCH_SIZE = 100                    # Количество названий в одном батче для эмбеддингов
MODEL_NAME = "gemini-1.5-flash"               # Модель для категоризации
EMBEDDING_MODEL = "models/text-embedding-004" # Модель для эмбеддингов
REQUEST_DELAY = 2                             # Задержка между запросами (сек)
MAX_RETRIES = 3                               # Максимальное число повторных попыток

# Настройки бота
ALLOWED_USERS = []                             # Список ID пользователей (пустой = все)
SEMSEARCH_TOP_K = 5                            # Количество результатов семантического поиска
SIMILAR_THEME_LIMIT = 10                       # Максимальное количество файлов той же темы для показа
# --------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Расширения видеофайлов
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}

# Инициализация бота
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Вычисляет косинусное сходство между двумя векторами."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class VideoProcessor:
    """Класс для обработки видео и работы с Gemini (категоризация)."""

    @staticmethod
    def get_video_filenames(folder: str) -> List[str]:
        """Возвращает список названий видеофайлов в указанной папке."""
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Папка '{folder}' не существует.")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"'{folder}' не является папкой.")

        video_files = []
        for item in folder_path.iterdir():
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(item.name)
        logger.info(f"Найдено видеофайлов: {len(video_files)}")
        return video_files

    @staticmethod
    def chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
        """Разбивает список на подсписки указанного размера."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    @staticmethod
    def build_prompt(filenames: List[str]) -> str:
        """Формирует текстовый промпт для нейросети."""
        file_list_str = "\n".join(f"- {name}" for name in filenames)
        prompt = f"""
Ты — помощник по категоризации видеоконтента. Ниже приведён список названий видеофайлов.
На основе этих названий придумай подходящие тематические группы (темы) и распредели файлы по этим темам.
Для каждого файла также предложи 2-4 ключевых слова (на русском языке), отражающих его содержание.

Верни ответ СТРОГО в формате JSON без лишнего текста. Структура должна быть такой:
{{
  "categories": [
    {{
      "theme": "Название темы 1",
      "files": [
        {{
          "filename": "имя_файла1.mp4",
          "keywords": ["ключевое", "слово", "ещё"]
        }},
        ...
      ]
    }},
    ...
  ]
}}

Названия файлов должны точно совпадать с исходными.

Список файлов:
{file_list_str}
"""
        return prompt

    @staticmethod
    def call_gemini_with_retries(prompt: str, api_key: str) -> Dict[str, Any]:
        """Отправляет запрос к Gemini с повторными попытками."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Отправка запроса к Gemini (попытка {attempt})...")
                response = model.generate_content(prompt)

                if not response.text:
                    raise ValueError("Пустой ответ от модели")

                raw_text = response.text.strip()

                # Убираем маркеры кода
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()

                result = json.loads(raw_text)

                if "categories" not in result:
                    raise ValueError("Ответ не содержит ключа 'categories'")

                return result

            except Exception as e:
                logger.warning(f"Ошибка при запросе: {e}")
                if attempt < MAX_RETRIES:
                    sleep_time = 2 ** attempt
                    logger.info(f"Повтор через {sleep_time} сек...")
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Не удалось получить ответ после {MAX_RETRIES} попыток.")

    @staticmethod
    def process_all_files(folder: str, api_key: str, batch_size: int) -> List[Dict[str, Any]]:
        """Обрабатывает все файлы и возвращает результаты."""
        filenames = VideoProcessor.get_video_filenames(folder)
        if not filenames:
            logger.warning("Нет видеофайлов для обработки.")
            return []

        all_results = []
        batches = list(VideoProcessor.chunk_list(filenames, batch_size))
        logger.info(f"Всего батчей: {len(batches)}")

        for idx, batch in enumerate(batches, 1):
            logger.info(f"Обработка батча {idx}/{len(batches)} ({len(batch)} файлов)")
            prompt = VideoProcessor.build_prompt(batch)

            try:
                response_data = VideoProcessor.call_gemini_with_retries(prompt, api_key)
            except Exception as e:
                logger.error(f"Критическая ошибка при обработке батча {idx}: {e}")
                for fname in batch:
                    all_results.append({
                        "filename": fname,
                        "theme": "ERROR",
                        "keywords": "Ошибка обработки"
                    })
                continue

            for category in response_data.get("categories", []):
                theme = category.get("theme", "Без темы")
                for file_info in category.get("files", []):
                    fname = file_info.get("filename", "")
                    keywords = ", ".join(file_info.get("keywords", []))
                    all_results.append({
                        "filename": fname,
                        "theme": theme,
                        "keywords": keywords
                    })

            if idx < len(batches):
                time.sleep(REQUEST_DELAY)

        return all_results

    @staticmethod
    def save_to_csv(results: List[Dict[str, Any]], output_path: str):
        """Сохраняет результаты в CSV файл."""
        if not results:
            logger.warning("Нет данных для сохранения.")
            return

        fieldnames = ["filename", "theme", "keywords"]
        with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Результаты сохранены в {output_path}")


class EmbeddingManager:
    """Класс для работы с эмбеддингами Gemini."""

    def __init__(self, api_key: str, model_name: str = EMBEDDING_MODEL):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.embeddings: Dict[str, np.ndarray] = {}  # filename -> embedding vector
        self.filenames: List[str] = []
        self.embedding_matrix: Optional[np.ndarray] = None

    def compute_embeddings(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[np.ndarray]:
        """
        Вычисляет эмбеддинги для списка текстов, используя батчи.
        Возвращает список numpy-векторов.
        """
        embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Вычисление эмбеддингов для батча {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} текстов)")

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # Используем batch_embed_contents для эффективности
                    result = genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type="retrieval_document"  # Для поиска документов
                    )
                    # result['embedding'] содержит список векторов
                    batch_embeddings = [np.array(emb) for emb in result['embedding']]
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    logger.warning(f"Ошибка при вычислении эмбеддингов: {e}")
                    if attempt < MAX_RETRIES:
                        sleep_time = 2 ** attempt
                        logger.info(f"Повтор через {sleep_time} сек...")
                        time.sleep(sleep_time)
                    else:
                        raise RuntimeError(f"Не удалось вычислить эмбеддинги после {MAX_RETRIES} попыток.")

            if i + batch_size < total:
                time.sleep(REQUEST_DELAY)  # Пауза между батчами

        return embeddings

    def build_from_filenames(self, filenames: List[str], force_recompute: bool = False):
        """
        Строит или загружает эмбеддинги для списка имен файлов.
        Если force_recompute=True, пересчитывает даже при наличии сохраненных.
        """
        if not force_recompute and os.path.exists(EMBEDDINGS_FILE):
            logger.info(f"Загрузка эмбеддингов из файла {EMBEDDINGS_FILE}")
            self.load(EMBEDDINGS_FILE)
            # Проверим, все ли файлы есть в загруженных данных
            missing = [f for f in filenames if f not in self.embeddings]
            if missing:
                logger.info(f"Отсутствуют эмбеддинги для {len(missing)} файлов, вычисляем...")
                new_embeddings = self.compute_embeddings(missing)
                for fname, emb in zip(missing, new_embeddings):
                    self.embeddings[fname] = emb
                self._update_matrix()
                self.save(EMBEDDINGS_FILE)
            else:
                self._update_matrix()
        else:
            logger.info(f"Вычисление эмбеддингов для {len(filenames)} файлов...")
            embeddings = self.compute_embeddings(filenames)
            self.embeddings = {fname: emb for fname, emb in zip(filenames, embeddings)}
            self._update_matrix()
            self.save(EMBEDDINGS_FILE)

    def _update_matrix(self):
        """Обновляет матрицу эмбеддингов и список имен файлов."""
        self.filenames = list(self.embeddings.keys())
        if self.filenames:
            self.embedding_matrix = np.vstack([self.embeddings[f] for f in self.filenames])
        else:
            self.embedding_matrix = None

    def save(self, filepath: str):
        """Сохраняет эмбеддинги в файл через pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.embeddings, f)
        logger.info(f"Эмбеддинги сохранены в {filepath}")

    def load(self, filepath: str):
        """Загружает эмбеддинги из файла pickle."""
        with open(filepath, 'rb') as f:
            self.embeddings = pickle.load(f)
        self._update_matrix()
        logger.info(f"Загружено {len(self.embeddings)} эмбеддингов")

    def search_similar(self, query: str, top_k: int = SEMSEARCH_TOP_K) -> List[Tuple[str, float]]:
        """
        Ищет top_k наиболее похожих файлов по семантическому сходству.
        Возвращает список кортежей (filename, similarity).
        """
        if self.embedding_matrix is None or len(self.filenames) == 0:
            return []

        # Получаем эмбеддинг запроса
        logger.info(f"Получение эмбеддинга для запроса: '{query}'")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=query,
                    task_type="retrieval_query"  # Для поискового запроса
                )
                query_embedding = np.array(result['embedding'])
                break
            except Exception as e:
                logger.warning(f"Ошибка при эмбеддинге запроса: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError("Не удалось получить эмбеддинг запроса")

        # Вычисляем косинусное сходство со всеми эмбеддингами
        # Нормализуем для эффективности
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        matrix_norm = self.embedding_matrix / np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        similarities = np.dot(matrix_norm, query_norm)

        # Получаем индексы top_k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.filenames[i], similarities[i]) for i in top_indices if similarities[i] > 0]

        return results

    def search_similar_by_filename(self, filename: str, top_k: int = SEMSEARCH_TOP_K) -> List[Tuple[str, float]]:
        """
        Ищет top_k наиболее похожих файлов, используя эмбеддинг указанного файла как запрос.
        """
        if filename not in self.embeddings:
            logger.warning(f"Эмбеддинг для файла {filename} не найден")
            return []
        
        query_embedding = self.embeddings[filename]
        
        # Вычисляем косинусное сходство со всеми эмбеддингами
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        matrix_norm = self.embedding_matrix / np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        similarities = np.dot(matrix_norm, query_norm)

        # Получаем индексы top_k (исключая сам файл)
        top_indices = np.argsort(similarities)[-top_k-1:][::-1]
        results = []
        for i in top_indices:
            if self.filenames[i] != filename and similarities[i] > 0:
                results.append((self.filenames[i], similarities[i]))
                if len(results) >= top_k:
                    break
        
        return results


class VideoSearcher:
    """Класс для поиска видео в CSV файле (по ключевым словам)."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data: List[Dict[str, str]] = []
        self.filename_to_row: Dict[str, Dict[str, str]] = {}
        self._load_data()

    def _load_data(self):
        """Загружает данные из CSV файла."""
        if not os.path.exists(self.csv_path):
            logger.warning(f"CSV файл {self.csv_path} не найден")
            return

        with open(self.csv_path, "r", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            self.data = list(reader)
            # Создаем индекс для быстрого поиска по имени файла
            for row in self.data:
                self.filename_to_row[row['filename']] = row
        logger.info(f"Загружено {len(self.data)} записей из CSV")

    def search(self, query: str) -> List[Dict[str, str]]:
        """Ищет query в названиях файлов и ключевых словах."""
        if not self.data:
            return []

        query_lower = query.lower().strip()
        results = []

        for row in self.data:
            filename = row.get("filename", "").lower()
            keywords = row.get("keywords", "").lower()

            if query_lower in filename or query_lower in keywords:
                results.append(row)

        return results

    def search_by_theme(self, theme: str) -> List[Dict[str, str]]:
        """Ищет все файлы с точным совпадением темы."""
        if not self.data:
            return []
        
        return [row for row in self.data if row.get("theme", "") == theme]

    def get_file_info(self, filename: str) -> Optional[Dict[str, str]]:
        """Возвращает информацию о файле по его имени."""
        return self.filename_to_row.get(filename)

    def format_results(self, results: List[Dict[str, str]], max_results: int = 50) -> Tuple[str, Optional[types.InlineKeyboardMarkup]]:
        """
        Форматирует результаты поиска для отображения в Telegram.
        Возвращает текст и inline-клавиатуру для первого результата.
        """
        if not results:
            return "❌ Ничего не найдено по вашему запросу.", None

        total = len(results)
        results = results[:max_results]

        message = f"🔍 Найдено результатов: {total}\n"
        if total > max_results:
            message += f"Показаны первые {max_results}:\n\n"
        else:
            message += "\n"

        # Создаем клавиатуру для первого результата
        keyboard = None
        if results:
            first_filename = results[0].get("filename", "")
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            btn_theme = types.InlineKeyboardButton(
                "📂 Похожие по теме", 
                callback_data=f"similar_theme:{first_filename}"
            )
            btn_semantic = types.InlineKeyboardButton(
                "🧠 Похожие по смыслу", 
                callback_data=f"similar_semantic:{first_filename}"
            )
            keyboard.add(btn_theme, btn_semantic)

        for i, row in enumerate(results, 1):
            filename = row.get("filename", "Без названия")
            theme = row.get("theme", "Без темы")
            keywords = row.get("keywords", "")

            message += f"{i}. 📹 {filename}\n"
            message += f"   📂 Тема: {theme}\n"
            if keywords:
                message += f"   🔑 Ключевые слова: {keywords}\n"
            message += "\n"

        message += "💡 Нажмите на кнопки ниже, чтобы найти похожие видео для первого результата."

        return message, keyboard

    def format_semantic_results(self, results: List[Tuple[str, float]], max_results: int = 5) -> Tuple[str, Optional[types.InlineKeyboardMarkup]]:
        """Форматирует результаты семантического поиска."""
        if not results:
            return "❌ Ничего не найдено.", None

        message = f"🧠 Семантический поиск (топ-{len(results)}):\n\n"
        
        # Создаем клавиатуру для первого результата
        keyboard = None
        if results:
            first_filename = results[0][0]
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            btn_theme = types.InlineKeyboardButton(
                "📂 Похожие по теме", 
                callback_data=f"similar_theme:{first_filename}"
            )
            btn_semantic = types.InlineKeyboardButton(
                "🧠 Похожие по смыслу", 
                callback_data=f"similar_semantic:{first_filename}"
            )
            keyboard.add(btn_theme, btn_semantic)

        for i, (filename, sim) in enumerate(results, 1):
            row = self.get_file_info(filename)
            theme = row.get("theme", "—") if row else "—"
            keywords = row.get("keywords", "") if row else ""

            message += f"{i}. 📹 {filename}\n"
            message += f"   📂 Тема: {theme}\n"
            if keywords:
                message += f"   🔑 Ключевые слова: {keywords}\n"
            message += f"   📊 Сходство: {sim:.3f}\n\n"

        message += "💡 Нажмите на кнопки ниже, чтобы найти похожие видео для первого результата."

        return message, keyboard

    def format_similar_theme_results(self, filename: str, theme: str, similar_files: List[Dict[str, str]]) -> str:
        """Форматирует результаты поиска файлов той же темы."""
        if not similar_files:
            return f"❌ Не найдено других файлов в теме '{theme}'."

        message = f"📂 Файлы в теме '{theme}' (всего {len(similar_files)}):\n\n"
        message += f"🔍 Исходный файл: {filename}\n\n"

        for i, row in enumerate(similar_files[:SIMILAR_THEME_LIMIT], 1):
            fname = row.get("filename", "")
            keywords = row.get("keywords", "")
            
            message += f"{i}. 📹 {fname}\n"
            if keywords:
                message += f"   🔑 {keywords}\n"
            message += "\n"

        if len(similar_files) > SIMILAR_THEME_LIMIT:
            message += f"... и ещё {len(similar_files) - SIMILAR_THEME_LIMIT} файлов"

        return message


# Глобальные объекты
searcher: Optional[VideoSearcher] = None
embedding_manager: Optional[EmbeddingManager] = None


def check_access(message: types.Message) -> bool:
    """Проверяет доступ пользователя к боту."""
    if not ALLOWED_USERS:
        return True
    return message.from_user.id in ALLOWED_USERS


# ---------------------- ОБРАБОТЧИКИ КОМАНД БОТА ----------------------

@bot.message_handler(commands=['start'])
def send_welcome(message: types.Message):
    if not check_access(message):
        bot.reply_to(message, "⛔ У вас нет доступа к этому боту.")
        return

    welcome_text = (
        "👋 Привет! Я бот для поиска видео по ключевым словам и семантическому смыслу.\n\n"
        "📋 Команды:\n"
        "/start - Показать приветствие\n"
        "/help - Показать справку\n"
        "/stats - Статистика базы видео\n"
        "/reload - Перезагрузить CSV файл\n"
        "/semsearch <запрос> - Семантический поиск по смыслу (например, /semsearch футбольный матч)\n\n"
        "🔍 Просто отправь мне ключевое слово, и я найду подходящие видео по тексту!\n"
        "Или используй /semsearch для поиска по смыслу.\n\n"
        "💡 После поиска появятся кнопки для нахождения похожих видео по теме или смыслу."
    )
    bot.reply_to(message, welcome_text)


@bot.message_handler(commands=['help'])
def send_help(message: types.Message):
    if not check_access(message):
        bot.reply_to(message, "⛔ У вас нет доступа к этому боту.")
        return

    help_text = (
        "📚 <b>Справка по использованию бота</b>\n\n"
        "<b>Обычный поиск:</b>\n"
        "Просто отправь ключевое слово или фразу.\n"
        "Бот ищет в названиях файлов и ключевых словах.\n\n"
        "<b>Семантический поиск:</b>\n"
        "Используй команду /semsearch с запросом.\n"
        "Пример: /semsearch обучение нейросетям\n"
        "Бот найдёт видео, близкие по смыслу, даже если точных слов нет в названии.\n\n"
        "<b>Интерактивные кнопки:</b>\n"
        "После поиска появляются кнопки:\n"
        "• 📂 Похожие по теме - показать другие видео той же темы\n"
        "• 🧠 Похожие по смыслу - найти семантически близкие видео\n\n"
        "<b>Доступные команды:</b>\n"
        "/start - Начало работы\n"
        "/help - Эта справка\n"
        "/stats - Количество видео в базе\n"
        "/reload - Обновить базу из CSV\n"
        "/semsearch - Семантический поиск\n\n"
        "Поиск не чувствителен к регистру."
    )
    bot.reply_to(message, help_text, parse_mode="HTML")


@bot.message_handler(commands=['stats'])
def send_stats(message: types.Message):
    if not check_access(message):
        bot.reply_to(message, "⛔ У вас нет доступа к этому боту.")
        return

    global searcher, embedding_manager
    if searcher is None:
        searcher = VideoSearcher(OUTPUT_CSV)

    total = len(searcher.data)
    if total == 0:
        bot.reply_to(message, "📊 База видео пуста. Запустите обработку видеофайлов.")
        return

    themes = {}
    for row in searcher.data:
        theme = row.get("theme", "Без темы")
        themes[theme] = themes.get(theme, 0) + 1

    stats_text = f"📊 <b>Статистика базы видео</b>\n\n"
    stats_text += f"📹 Всего видео: <b>{total}</b>\n"
    stats_text += f"📂 Количество тем: <b>{len(themes)}</b>\n"

    # Информация об эмбеддингах
    if embedding_manager and embedding_manager.embeddings:
        stats_text += f"🧠 Эмбеддинги: загружено {len(embedding_manager.embeddings)} векторов\n"
    else:
        stats_text += f"🧠 Эмбеддинги: не загружены\n"

    stats_text += "\n<b>Топ-10 тем:</b>\n"
    sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (theme, count) in enumerate(sorted_themes, 1):
        stats_text += f"{i}. {theme}: {count}\n"

    bot.reply_to(message, stats_text, parse_mode="HTML")


@bot.message_handler(commands=['reload'])
def reload_csv(message: types.Message):
    if not check_access(message):
        bot.reply_to(message, "⛔ У вас нет доступа к этому боту.")
        return

    global searcher
    try:
        searcher = VideoSearcher(OUTPUT_CSV)
        total = len(searcher.data)
        bot.reply_to(message, f"✅ База перезагружена! Загружено {total} записей.")
    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при перезагрузке: {str(e)}")


@bot.message_handler(commands=['semsearch'])
def handle_semantic_search(message: types.Message):
    """Обработчик семантического поиска."""
    if not check_access(message):
        bot.reply_to(message, "⛔ У вас нет доступа к этому боту.")
        return

    global embedding_manager, searcher
    if searcher is None:
        searcher = VideoSearcher(OUTPUT_CSV)

    if not searcher.data:
        bot.reply_to(message, "⚠️ База видео пуста. Сначала обработайте видеофайлы.")
        return

    # Извлекаем запрос (текст после команды)
    query = message.text.strip()
    if query.startswith('/semsearch'):
        query = query[len('/semsearch'):].strip()
    if not query:
        bot.reply_to(message, "⚠️ Укажите запрос после команды, например: /semsearch футбольный матч")
        return

    # Проверяем наличие эмбеддингов
    if embedding_manager is None:
        embedding_manager = EmbeddingManager(GEMINI_API_KEY)

    if not embedding_manager.embeddings:
        # Нужно построить эмбеддинги
        bot.reply_to(message, "⏳ Эмбеддинги ещё не построены. Начинаю вычисление, это может занять несколько минут...")
        try:
            filenames = [row['filename'] for row in searcher.data]
            embedding_manager.build_from_filenames(filenames)
            bot.reply_to(message, f"✅ Эмбеддинги построены для {len(embedding_manager.embeddings)} файлов. Выполняю поиск...")
        except Exception as e:
            bot.reply_to(message, f"❌ Ошибка при построении эмбеддингов: {str(e)}")
            return

    # Выполняем поиск
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        results = embedding_manager.search_similar(query, top_k=SEMSEARCH_TOP_K)
        response, keyboard = searcher.format_semantic_results(results)
        
        if keyboard:
            bot.reply_to(message, response, reply_markup=keyboard)
        else:
            bot.reply_to(message, response)
    except Exception as e:
        logger.error(f"Ошибка семантического поиска: {e}")
        bot.reply_to(message, f"❌ Ошибка при поиске: {str(e)}")


@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call: types.CallbackQuery):
    """Обработчик inline-кнопок."""
    global searcher, embedding_manager
    
    if not check_access(call.message):
        bot.answer_callback_query(call.id, "⛔ У вас нет доступа.")
        return

    try:
        # Парсим callback_data
        if call.data.startswith("similar_theme:"):
            filename = call.data.split(":", 1)[1]
            
            # Получаем информацию о файле
            file_info = searcher.get_file_info(filename)
            if not file_info:
                bot.answer_callback_query(call.id, "❌ Файл не найден в базе")
                return
            
            theme = file_info.get("theme", "")
            if not theme:
                bot.answer_callback_query(call.id, "❌ У файла не указана тема")
                return
            
            # Ищем файлы той же темы
            similar_files = searcher.search_by_theme(theme)
            # Убираем исходный файл из результатов
            similar_files = [f for f in similar_files if f.get("filename") != filename]
            
            response = searcher.format_similar_theme_results(filename, theme, similar_files)
            bot.answer_callback_query(call.id, "✅ Найдены похожие файлы")
            bot.send_message(call.message.chat.id, response)
            
        elif call.data.startswith("similar_semantic:"):
            filename = call.data.split(":", 1)[1]
            
            # Проверяем наличие эмбеддингов
            if embedding_manager is None:
                embedding_manager = EmbeddingManager(GEMINI_API_KEY)
            
            if not embedding_manager.embeddings:
                bot.answer_callback_query(call.id, "⏳ Загружаю эмбеддинги...")
                # Пытаемся загрузить или построить эмбеддинги
                if os.path.exists(EMBEDDINGS_FILE):
                    embedding_manager.load(EMBEDDINGS_FILE)
                else:
                    bot.send_message(call.message.chat.id, "⏳ Эмбеддинги ещё не построены. Начинаю вычисление...")
                    filenames = [row['filename'] for row in searcher.data]
                    embedding_manager.build_from_filenames(filenames)
            
            # Проверяем, есть ли эмбеддинг для файла
            if filename not in embedding_manager.embeddings:
                bot.answer_callback_query(call.id, "❌ Эмбеддинг для файла не найден")
                return
            
            bot.answer_callback_query(call.id, "🔍 Ищу семантически похожие видео...")
            bot.send_chat_action(call.message.chat.id, 'typing')
            
            # Ищем похожие файлы
            results = embedding_manager.search_similar_by_filename(filename, top_k=SEMSEARCH_TOP_K)
            
            if not results:
                bot.send_message(call.message.chat.id, "❌ Не найдено похожих видео")
                return
            
            # Форматируем результаты
            message = f"🧠 Видео, похожие по смыслу на '{filename}':\n\n"
            for i, (fname, sim) in enumerate(results, 1):
                row = searcher.get_file_info(fname)
                theme = row.get("theme", "—") if row else "—"
                keywords = row.get("keywords", "") if row else ""
                
                message += f"{i}. 📹 {fname}\n"
                message += f"   📂 Тема: {theme}\n"
                if keywords:
                    message += f"   🔑 {keywords}\n"
                message += f"   📊 Сходство: {sim:.3f}\n\n"
            
            bot.send_message(call.message.chat.id, message)
            
    except Exception as e:
        logger.error(f"Ошибка в callback handler: {e}")
        bot.answer_callback_query(call.id, f"❌ Ошибка: {str(e)}")


@bot.message_handler(func=lambda message: True)
def handle_search(message: types.Message):
    """Обрабатывает обычные текстовые поисковые запросы."""
    if not check_access(message):
        bot.reply_to(message, "⛔ У вас нет доступа к этому боту.")
        return

    global searcher
    if searcher is None:
        searcher = VideoSearcher(OUTPUT_CSV)

    if not searcher.data:
        bot.reply_to(
            message,
            "⚠️ База видео пуста. Сначала запустите скрипт для обработки видеофайлов."
        )
        return

    query = message.text.strip()
    if len(query) < 2:
        bot.reply_to(message, "⚠️ Слишком короткий запрос. Введите минимум 2 символа.")
        return

    bot.send_chat_action(message.chat.id, 'typing')
    logger.info(f"Поиск по запросу: '{query}' от пользователя {message.from_user.id}")

    results = searcher.search(query)
    response, keyboard = searcher.format_results(results)

    # Если ответ слишком длинный, разбиваем на части (без клавиатуры для длинных ответов)
    if len(response) > 4096:
        for i in range(0, len(response), 4096):
            bot.reply_to(message, response[i:i+4096])
    else:
        if keyboard:
            bot.reply_to(message, response, reply_markup=keyboard)
        else:
            bot.reply_to(message, response)


def run_bot():
    """Запускает Telegram бота."""
    logger.info("Запуск Telegram бота...")
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        logger.error(f"Ошибка в работе бота: {e}")
        time.sleep(10)
        run_bot()


def process_videos_and_save():
    """Обрабатывает видео через Gemini и сохраняет в CSV."""
    logger.info("Начинаем обработку видеофайлов...")
    try:
        results = VideoProcessor.process_all_files(FOLDER_PATH, GEMINI_API_KEY, BATCH_SIZE)
        VideoProcessor.save_to_csv(results, OUTPUT_CSV)
        logger.info("Обработка видео завершена!")
        return True
    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {e}")
        return False


def compute_embeddings_only():
    """Вычисляет и сохраняет эмбеддинги на основе существующего CSV."""
    global searcher
    searcher = VideoSearcher(OUTPUT_CSV)
    if not searcher.data:
        logger.error("CSV файл пуст или не найден. Сначала обработайте видео.")
        return False

    filenames = [row['filename'] for row in searcher.data]
    global embedding_manager
    embedding_manager = EmbeddingManager(GEMINI_API_KEY)
    try:
        embedding_manager.build_from_filenames(filenames, force_recompute=True)
        logger.info(f"Эмбеддинги вычислены и сохранены в {EMBEDDINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при вычислении эмбеддингов: {e}")
        return False


def main():
    """Главная функция."""
    try:
        # Проверка API ключей
        if GEMINI_API_KEY == "ВАШ_API_КЛЮЧ_GEMINI":
            logger.error("❌ Не задан API-ключ Gemini. Отредактируйте GEMINI_API_KEY.")
            return

        if TELEGRAM_BOT_TOKEN == "ВАШ_ТОКЕН_БОТА":
            logger.error("❌ Не задан токен Telegram бота. Отредактируйте TELEGRAM_BOT_TOKEN.")
            return

        # Проверка папки
        if not os.path.isdir(FOLDER_PATH):
            logger.error(f"❌ Папка '{FOLDER_PATH}' не найдена. Проверьте FOLDER_PATH.")
            return

        print("\n" + "="*50)
        print("🤖 VIDEO SEARCH BOT WITH GEMINI + EMBEDDINGS")
        print("="*50)
        print("\nВыберите режим работы:")
        print("1. Только обработать видео (создать/обновить CSV)")
        print("2. Только запустить бота (использовать существующие CSV и эмбеддинги)")
        print("3. Обработать видео И запустить бота")
        print("4. Вычислить/обновить эмбеддинги для семантического поиска")
        print("\n" + "="*50)

        choice = input("\nВаш выбор (1/2/3/4): ").strip()

        if choice == "1":
            if process_videos_and_save():
                print("\n✅ Обработка завершена! CSV файл создан.")
            else:
                print("\n❌ Ошибка при обработке видео.")

        elif choice == "2":
            if not os.path.exists(OUTPUT_CSV):
                print(f"\n⚠️ CSV файл '{OUTPUT_CSV}' не найден. Сначала обработайте видео.")
                return

            global searcher, embedding_manager
            searcher = VideoSearcher(OUTPUT_CSV)
            print(f"\n✅ Загружено {len(searcher.data)} записей из CSV.")

            # Попробуем загрузить эмбеддинги, если есть
            if os.path.exists(EMBEDDINGS_FILE):
                embedding_manager = EmbeddingManager(GEMINI_API_KEY)
                embedding_manager.load(EMBEDDINGS_FILE)
                print(f"✅ Загружено {len(embedding_manager.embeddings)} эмбеддингов.")
            else:
                print("⚠️ Файл эмбеддингов не найден. Семантический поиск будет недоступен до вычисления.")
                embedding_manager = EmbeddingManager(GEMINI_API_KEY)

            print("\n🤖 Запуск бота...")
            run_bot()

        elif choice == "3":
            print("\n🔄 Обработка видео...")
            if process_videos_and_save():
                print("✅ Обработка завершена!")
            else:
                print("⚠️ Ошибка при обработке, но продолжаем с существующими данными...")

            global searcher, embedding_manager
            searcher = VideoSearcher(OUTPUT_CSV)
            print(f"\n✅ Загружено {len(searcher.data)} записей из CSV.")

            if os.path.exists(EMBEDDINGS_FILE):
                embedding_manager = EmbeddingManager(GEMINI_API_KEY)
                embedding_manager.load(EMBEDDINGS_FILE)
                print(f"✅ Загружено {len(embedding_manager.embeddings)} эмбеддингов.")
            else:
                print("⚠️ Файл эмбеддингов не найден. Семантический поиск будет недоступен до вычисления.")
                embedding_manager = EmbeddingManager(GEMINI_API_KEY)

            print("\n🤖 Запуск бота...")
            run_bot()

        elif choice == "4":
            if not os.path.exists(OUTPUT_CSV):
                print(f"\n⚠️ CSV файл '{OUTPUT_CSV}' не найден. Сначала обработайте видео.")
                return
            print("\n🧠 Вычисление эмбеддингов...")
            if compute_embeddings_only():
                print("\n✅ Эмбеддинги успешно вычислены и сохранены!")
            else:
                print("\n❌ Ошибка при вычислении эмбеддингов.")

        else:
            print("❌ Неверный выбор. Запустите скрипт снова.")

    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
    except Exception as e:
        logger.exception(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()
