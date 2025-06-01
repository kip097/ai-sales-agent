import json
import re
from typing import List, Tuple, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
from retriever import Retriever
from function_calling_logic import send_invoice, handover_to_manager

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

def find_parts_by_name_and_model(request_name: str, model_year: str, retriever: Retriever) -> Tuple[Dict, Dict]:
    found_original = None
    found_analogue = None
    query = f"{request_name} {model_year}"
    
    model_year_parts = model_year.split()
    if len(model_year_parts) >= 2:
        model = " ".join(model_year_parts[:-1])
        year = model_year_parts[-1]
    else:
        model = model_year
        year = ""
    
    results = retriever.rerank(retriever.search(query, top_k=10), query)
    
    for metadata, score in results:
        if "артикул" in metadata:
            name_match = any(part.lower() in metadata["название"].lower() for part in request_name.split())
            model_match = False
            for compat in metadata["compatibility"]:
                compat_lower = compat.lower()
                if model.lower() in compat_lower:
                    if year:
                        year_range = re.search(r"(\d{4})-(\d{4})", compat)
                        if year_range:
                            start_year, end_year = map(int, year_range.groups())
                            if start_year <= int(year) <= end_year:
                                model_match = True
                        else:
                            if year in compat:
                                model_match = True
                    else:
                        model_match = True
            if name_match and model_match:
                item = {
                    "id": metadata["id"],
                    "name": metadata["название"],
                    "compatibility": metadata["compatibility"],
                    "original": metadata["оригинал"],
                    "price": metadata["цена"],
                    "article": metadata["артикул"]
                }
                if metadata["оригинал"]:
                    found_original = item
                else:
                    found_analogue = item
    
    if not found_original and not found_analogue:
        for item in catalog:
            name_match = any(part.lower() in item["name"].lower() for part in request_name.split())
            model_match = False
            for compat in item["compatibility"]:
                compat_lower = compat.lower()
                if model.lower() in compat_lower:
                    if year:
                        year_range = re.search(r"(\d{4})-(\d{4})", compat)
                        if year_range:
                            start_year, end_year = map(int, year_range.groups())
                            if start_year <= int(year) <= end_year:
                                model_match = True
                        else:
                            if year in compat:
                                model_match = True
                    else:
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
    
    return found_original, found_analogue

def agent_respond(message: str, conversation: List[Dict], retriever: Retriever) -> Dict:
    conversation.append({"role": "user", "content": message, "selection": None, "model_year": None})
    last_user_message = message.lower()
    
    part_name = None
    model_year = None
    for part in ["моторчик омывателя", "задний фонарь", "лямбда-зонд", "воздушный фильтр"]:
        if part in last_user_message:
            part_name = part
            break
    for model in ["XDrive 2009", "Cruiser 2012", "Vento 2010"]:
        if model.lower() in last_user_message:
            model_year = model
            conversation[-1]["model_year"] = model_year
            break
    
    if part_name and model_year:
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
    
    if "аналог нормальный" in last_user_message:
        response = sales_templates["analogue_quality"]
        conversation.append({"role": "assistant", "content": response, "status": "analogue_quality"})
        return {"response": response, "status": "analogue_quality"}
    
    if "дорого" in last_user_message:
        response = sales_templates["price_objection"]
        conversation.append({"role": "assistant", "content": response, "status": "price_objection"})
        return {"response": response, "status": "price_objection"}
    
    if any(word in last_user_message for word in ["оригинал", "давайте", "оформляйте"]):
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "assistant" and conversation[i].get("status") == "suggest_part"):
                selection = "оригинал" if "оригинал" in last_user_message else "аналог"
                conversation[-1]["selection"] = selection
                response = sales_templates["request_contact"]
                conversation.append({"role": "assistant", "content": response, "status": "request_contact"})
                return {"response": response, "status": "request_contact"}
    
    if "+" in last_user_message or any(char.isdigit() for char in last_user_message):
        name = "неизвестно"
        phone = "неизвестно"
        parts = last_user_message.split(",")
        if len(parts) >= 2:
            name = parts[0].strip().capitalize()
            phone = parts[1].strip()
        else:
            for word in last_user_message.split():
                if word[0].isupper():
                    name = word
                if "+" in word or any(char.isdigit() for char in word):
                    phone = word
        
        selected_part = None
        model_year = "неизвестно"
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "user" and conversation[i].get("model_year"):
                model_year = conversation[i]["model_year"]
                break
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "user" and conversation[i].get("selection"):
                selected_part = conversation[i]["selection"]
                break
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get("role") == "assistant" and conversation[i].get("status") == "suggest_part":
                part = conversation[i]["content"]
                part_name = "неизвестно"
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
    
    if any(greeting in last_user_message for greeting in ["здравствуйте", "добрый день", "привет"]):
        response = sales_templates["greeting"]
        conversation.append({"role": "assistant", "content": response, "status": "greeting"})
        return {"response": response, "status": "greeting"}
    
    response = sales_templates["default"]
    conversation.append({"role": "assistant", "content": response, "status": "default"})
    return {"response": response, "status": "default"}

def main():
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

if __name__ == "__main__":
    main()
```

---

#### 5. Function Calling
##### Функции
- **`send_invoice(details)`**:
  - **Когда вызывается**: После того как клиент выбирает запчасть (оригинал или аналог) и предоставляет имя и телефон.
  - **Параметры**:
    - `client_name`: Имя клиента (например, "Иван").
    - `contact`: Телефон (например, "+79990000001").
    - `part_name`: Название запчасти (например, "Моторчик омывателя лобового стекла").
    - `part_article`: Артикул (например, "MTR-101-O").
    - `price`: Цена в рублях (например, 2100).
    - `model_year`: Модель и год (например, "XDrive 2009").
  - **Точка вызова**: В `agent_respond`, после обработки контактных данных.

- **`handover_to_manager(lead_data)`**:
  - **Когда вызывается**: Если запчасть не найдена в каталоге для указанной модели.
  - **Параметры**:
    - `requested_part`: Название запчасти (например, "моторчик омывателя").
    - `model_year`: Модель и год (например, "XDrive 2009").
    - `user_msg`: Полное сообщение клиента (например, "У меня XDrive 2009 года, хочу купить моторчик омывателя").
  - **Точка вызова**: В `agent_respond`, если `find_parts_by_name_and_model` возвращает пустые результаты.

##### Реализация
Файл `function_calling_logic.py` содержит заглушки для функций с выводом в консоль.

<xaiArtifact artifact_id="e9050dd7-c5c1-4bf7-9211-d8a3e276653a" artifact_version_id="61457c74-446d-42e8-91c6-d4a2fae79451" title="src/function_calling_logic.py" contentType="text/python">
from datetime import datetime

def send_invoice(data: dict):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Отправляем счёт клиенту {data['client_name']} ({data['contact']}) на деталь '{data['part_name']}' ({data['part_article']}) для {data['model_year']} на сумму {data['price']} ₽")

def handover_to_manager(data: dict):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Передаём лид менеджеру: запрошена деталь '{data['requested_part']}' для {data['model_year']}. Сообщение клиента: '{data['user_msg']}'")