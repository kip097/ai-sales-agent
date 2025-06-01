import json
import re
import os
from typing import List, Tuple, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
from retriever import Retriever
from function_calling_logic import send_invoice, handover_to_manager

# Проверка наличия файлов данных
def check_files_exist():
    required_files = ["data/sales_templates.json", "data/catalog_chunks.json"]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден в директории {os.getcwd()}.")

# Загрузка данных с обработкой исключений
try:
    check_files_exist()
    with open("data/sales_templates.json", encoding="utf-8") as f:
        templates_data = json.load(f)
        sales_templates = {item["key"]: item["value"] for item in templates_data}

    with open("data/catalog_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
        catalog = [
            {
                "id": c["metadata"]["id"],
                "name": c["metadata"]["название"],
                "compatibility": c["metadata"]["compatibility"],
                "original": c["metadata"]["оригинал"],
                "price": c["metadata"]["цена"],
                "article": c["metadata"]["артикул"]
            }
            for c in chunks if "артикул" in c["metadata"]
        ]
    # Отладочный вывод: проверяем содержимое каталога
    print("Загруженный каталог:", [item["name"] + " (" + item["article"] + ")" for item in catalog])
except FileNotFoundError as e:
    print(f"Ошибка: {e}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Ошибка при разборе JSON: {e}")
    exit(1)

def find_parts_by_name_and_model(request_name: str, model_year: str, retriever: Retriever) -> Tuple[Dict, Dict]:
    found_original = None
    found_analogue = None
    query = f"{request_name} {model_year}"

    model_year_parts = model_year.split()
    if len(model_year_parts) >= 2:
        model = " ".join(model_year_parts[:-1]).lower()
        year = model_year_parts[-1]
    else:
        model = model_year.lower()
        year = ""

    for item in catalog:
        name_match = request_name.lower() in item["name"].lower()
        model_match = False
        for compat in item["compatibility"]:
            compat_lower = compat.lower()

            if model in compat_lower:
                if re.search(r"\d{4}[\u2013-]\d{4}", compat):
                    # диапазон лет
                    year_range = re.search(r"(\d{4})[\u2013-](\d{4})", compat)
                    if year_range and year.isdigit():
                        start_year, end_year = map(int, year_range.groups())
                        if start_year <= int(year) <= end_year:
                            model_match = True
                    else:
                        model_match = True
                else:
                    # нет диапазона — считаем совместимым по модели
                    model_match = True

        if name_match and model_match:
            new_item = {
                "id": item["id"],
                "name": item["name"],
                "compatibility": item["compatibility"],
                "original": item["original"],
                "price": item["price"],
                "article": item["article"]
            }
            if item["original"]:
                found_original = new_item
            else:
                found_analogue = new_item

    print(f"Поиск: запрос='{query}', найдено: оригинал={found_original}, аналог={found_analogue}")
    return found_original, found_analogue


def agent_respond(message: str, conversation: List[Dict], retriever: Retriever) -> Dict:
    conversation.append({"role": "user", "content": message, "selection": None, "model_year": None, "part_name": None})
    last_user_message = message.lower().strip()
    
    # Определение запчасти и модели
    part_name = None
    model_year = None
    for part in ["моторчик омывателя", "задний фонарь", "лямбда-зонд", "воздушный фильтр"]:
        if part in last_user_message:
            part_name = part
            conversation[-1]["part_name"] = part_name
            break
    for model in ["xdrive 2009", "cruiser 2012", "vento 2010"]:
        if model in last_user_message:
            model_year = model
            conversation[-1]["model_year"] = model_year
            break
    
    # Предложение запчастей
    if part_name and model_year and not any(c["status"] == "not_found" for c in conversation if "status" in c):
        original, analogue = find_parts_by_name_and_model(part_name, model_year, retriever)
        if original or analogue:
            response = sales_templates["suggest_part"].format(
                model_year=model_year,
                part_name=part_name,
                original_name=original["name"] if original else "нет",
                original_article=original["article"] if original else "нет",
                original_price=original["price"] if original else "нет",
                analogue_name=analogue["name"] if analogue else "нет",
                analogue_article=analogue["article"] if analogue else "нет",
                analogue_price=analogue["price"] if analogue else "нет"
            )
            conversation.append({"role": "assistant", "content": response, "status": "suggest_part"})
            return {"response": response, "status": "suggest_part"}
        else:
            context = f"Запрошенная деталь: {part_name}, модель и год: {model_year}"
            handover_to_manager({"requested_part": part_name, "model_year": model_year, "user_msg": last_user_message})
            response = sales_templates["not_found"]
            conversation.append({"role": "assistant", "content": response, "status": "not_found"})
            return {"response": response, "status": "not_found"}
    
    # Обработка вопросов о качестве аналога
    if "аналог нормальный" in last_user_message:
        response = sales_templates["analogue_quality"]
        conversation.append({"role": "assistant", "content": response, "status": "analogue_quality"})
        return {"response": response, "status": "analogue_quality"}
    
    # Обработка возражений по цене
    if "дорого" in last_user_message:
        response = sales_templates["price_objection"]
        conversation.append({"role": "assistant", "content": response, "status": "price_objection"})
        return {"response": response, "status": "price_objection"}
    
    # Выбор запчасти
    if any(word in last_user_message for word in ["оригинал", "давайте", "оформляйте"]):
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "assistant" and conversation[i].get("status") == "suggest_part":
                selection = "оригинал" if "оригинал" in last_user_message else "аналог"
                conversation[-1]["selection"] = selection
                response = sales_templates["request_contact"]
                conversation.append({"role": "assistant", "content": response, "status": "request_contact"})
                return {"response": response, "status": "request_contact"}
    
    # Обработка контактных данных
    if "+" in last_user_message or any(char.isdigit() for char in last_user_message):
        name = "неизвестно"
        phone = "неизвестно"
        parts = [p.strip() for p in last_user_message.split(",") if p.strip()]
        if len(parts) >= 2:
            name = parts[0].capitalize()
            phone = parts[1]
        else:
            words = last_user_message.split()
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    name = word
                if "+" in word or word.isdigit():
                    phone = word
        
        selected_part = None
        model_year = "неизвестно"
        part_name = "неизвестно"
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "user" and conversation[i].get("model_year"):
                model_year = conversation[i]["model_year"]
                break
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "user" and conversation[i].get("part_name"):
                part_name = conversation[i]["part_name"]
                break
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "user" and conversation[i].get("selection"):
                selected_part = conversation[i]["selection"]
                break
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "assistant" and conversation[i].get("status") == "suggest_part":
                part = conversation[i]["content"]
                article = "неизвестно"
                price = 0
                if selected_part == "оригинал":
                    if "(MTR-101-O)" in part:
                        part_name = "Моторчик омывателя лобового стекла"
                        article = "MTR-101-O"
                        price = 2100
                    elif "(LGT-201-O)" in part:
                        part_name = "Задний фонарь"
                        article = "LGT-201-O"
                        price = 5900
                    elif "(LMB-301-O)" in part:
                        part_name = "Лямбда-зонд"
                        article = "LMB-301-O"
                        price = 2800
                    elif "(FLT-401-O)" in part:
                        part_name = "Воздушный фильтр двигателя"
                        article = "FLT-401-O"
                        price = 800
                elif selected_part == "аналог":
                    if "(MTR-101-A)" in part:
                        part_name = "Моторчик омывателя (аналог)"
                        article = "MTR-101-A"
                        price = 1350
                    elif "(LGT-201-D)" in part:
                        part_name = "Задний фонарь (аналог Depo)"
                        article = "LGT-201-D"
                        price = 3600
                    elif "(LMB-301-U)" in part:
                        part_name = "Лямбда-зонд универсальный"
                        article = "LMB-301-U"
                        price = 1900
                    elif "(FLT-401-A)" in part:
                        part_name = "Воздушный фильтр (аналог)"
                        article = "FLT-401-A"
                        price = 500
                if part_name != "неизвестно":
                    send_invoice({
                        "client_name": name,
                        "contact": phone,
                        "part_name": part_name,
                        "part_article": article,
                        "price": price,
                        "model_year": model_year
                    })
                    response = sales_templates["thank_you"]
                    conversation.append({"role": "assistant", "content": response, "status": "order_placed"})
                    return {"response": response, "status": "order_placed"}
        # Если запчасть не найдена ранее
        response = sales_templates["default"]
        conversation.append({"role": "assistant", "content": response, "status": "default"})
        return {"response": response, "status": "default"}
    
    # Обработка приветствия
    if any(greeting in last_user_message for greeting in ["здравствуйте", "добрый день", "привет"]):
        response = sales_templates["greeting"]
        conversation.append({"role": "assistant", "content": response, "status": "greeting"})
        return {"response": response, "status": "greeting"}
    
    # Ответ по умолчанию
    response = sales_templates["default"]
    conversation.append({"role": "assistant", "content": response, "status": "default"})
    return {"response": response, "status": "default"}

def main():
    try:
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        retriever = Retriever(embedder_model_name="paraphrase-MiniLM-L6-v2")
        retriever.build_index(chunks)

        # Диалог 1
        conversation = []
        messages = [
            "Здравствуйте!",
            "У меня XDrive 2009 года, хочу купить моторчик омывателя.",
            "Давайте оригинал.",
            "Иван, +79990000001"
        ]
        print("\n--- Диалог 1 ---")
        for msg in messages:
            print(f"Клиент: {msg}")
            response = agent_respond(msg, conversation, retriever)
            print(f"Агент: {response['response']}")

        # Диалог 2
        conversation = []
        messages = [
            "Добрый день.",
            "Cruiser 2012 года, интересует задний фонарь.",
            "Аналог нормальный?",
            "Да, оформляйте.",
            "Петр, +79990000002"
        ]
        print("\n--- Диалог 2 ---")
        for msg in messages:
            print(f"Клиент: {msg}")
            response = agent_respond(msg, conversation, retriever)
            print(f"Агент: {response['response']}")

        # Диалог 3
        conversation = []
        messages = [
            "Привет!",
            "Vento 2010, лямбда-зонд нужен.",
            "Слишком дорого, есть дешевле?",
            "Хорошо, давайте аналог.",
            "Анна, +79990000003"
        ]
        print("\n--- Диалог 3 ---")
        for msg in messages:
            print(f"Клиент: {msg}")
            response = agent_respond(msg, conversation, retriever)
            print(f"Агент: {response['response']}")
    except Exception as e:
        print(f"Ошибка при выполнении: {e}")
        exit(1)

if __name__ == "__main__":
    main()
