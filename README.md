# AI Sales Agent

Чат-бот для продажи автозапчастей с использованием RAG и Sentence Transformers.

## Установка
```bash
conda create -n ai_sales_env python=3.10
conda activate ai_sales_env
conda install -c conda-forge faiss-cpu=1.7.4 sentencepiece=0.1.99 numpy=1.26.4 tqdm
pip install sentence-transformers==2.2.2 huggingface_hub==0.16.4 tokenizers==0.13.3 transformers==4.30.2
```

## Запуск
```bash
python3 src/main.py
```

## Структура
- `src/main.py`: Основной скрипт с логикой диалогов.
- `src/retriever.py`: Модуль для поиска с использованием FAISS.
- `src/function_calling_logic.py`: Функции для отправки счетов и передачи менеджеру.
- `data/catalog_chunks.json`: Каталог запчастей.
- `data/sales_templates.json`: Шаблоны ответов.

## Диалоги
- Запрашивает деталь и модель, предлагает оригинал и аналог.
- Оформляет заказ с указанием имени и телефона.
- Передаёт сложные запросы менеджеру.
