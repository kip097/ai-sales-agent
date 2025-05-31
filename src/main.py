import json
import re
import uuid
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import faiss
from retriever import Retriever
from function_calling_logic import send_invoice, handover_to_manager

# Загрузка данных
with open("data/catalog_chunks.json", "r", encoding="utf-8") as f:
    catalog_chunks = json.load(f)

with open("data/sales_templates.json", "r", encoding="utf-8") as f:
    sales_templates = json.load(f)

# Формирование каталога из catalog_chunks.json
catalog = []
for chunk in catalog_chunks:
    catalog.append({
        "id": chunk["metadata"]["id"],
        "name": chunk["metadata"]["название"],
        "compatibility": chunk["metadata"]["compatibility"],
        "original": chunk["metadata"]["оригинал"],
        "price": chunk["metadata"]["цена"],
        "article": chunk["metadata"]["артикул"]
    })

# Формирование фраз из sales_templates.json
sales_phrases = {}
for template in sales_templates:
    situation = template["metadata"]["situation"]
    for phrase in template["metadata"]["phrases"]:
        sales_phrases[f"{situation}_{uuid.uuid4().hex[:8]}"] = phrase

# Вспомогательная функция поиска деталей по запросу и модели
def find_parts_by_name_and_model(request_name: str, model_year: str, retriever: Retriever) -> Tuple[Dict, Dict]:
    found_original = None
    found_analogue = None
    query = f"{request_name} {model_year}"
    results = retriever.rerank(retriever.search(query, top_k=5), query)  # Увеличиваем top_k
    for metadata, _ in results:
        if "артикул" in metadata and request_name.lower() in metadata["название"].lower():
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
    return found_original, found_analogue

# Логика диалога
def agent_respond(history: List[Tuple[str, str]], user_msg: str, client_data: Dict, retriever: Retriever) -> str:
    if client_data.get("state") == "start":
        client_data["state"] = "wait_model_year"
        results = retriever.search("greeting", top_k=1)
        return results[0][0]["phrases"][0] if results and "phrases" in results[0][0] else "Добрый день! Чем могу помочь?"

    if client_data.get("state") == "wait_model_year":
        # Улучшенный парсинг модели и года
        year_match = re.search(r"(19[9]\d|20[0-2]\d|2025)", user_msg)
        model_match = None
        words = user_msg.split()
        for i, w in enumerate(words):
            if w[0].isupper() and not w.isdigit():
                # Проверяем, является ли слово моделью (например, XDrive, Cruiser)
                model_match = w
                if i > 0 and words[i-1].lower() in ["для", "на"]:  # Контекст: "для XDrive"
                    model_match = words[i-1] + " " + w
                break

        if year_match and model_match:
            model_year = f"{model_match} {year_match.group()}"
            client_data["model_year"] = model_year
            client_data["state"] = "offer_part"

            orig, anal = find_parts_by_name_and_model(client_data["requested_part"], model_year, retriever)
            client_data["selected_original"] = orig
            client_data["selected_analogue"] = anal

            if orig and anal:
                results = retriever.search("offer_parts", top_k=1)
                return results[0][0]["phrases"][0].format(
                    model_year=model_year,
                    orig_article=orig["article"],
                    orig_price=orig["price"],
                    analogue_article=anal["article"],
                    analogue_price=anal["price"]
                ) if results else sales_phrases["offer_parts_12345678"].format(
                    model_year=model_year,
                    orig_article=orig["article"],
                    orig_price=orig["price"],
                    analogue_article=anal["article"],
                    analogue_price=anal["price"]
                )
            elif orig:
                results = retriever.search("offer_only_original", top_k=1)
                return results[0][0]["phrases"][0].format(
                    model_year=model_year,
                    orig_article=orig["article"],
                    orig_price=orig["price"]
                ) if results else sales_phrases["offer_only_original_12345678"].format(
                    model_year=model_year,
                    orig_article=orig["article"],
                    orig_price=orig["price"]
                )
            elif anal:
                results = retriever.search("offer_only_analogue", top_k=1)
                return results[0][0]["phrases"][0].format(
                    model_year=model_year,
                    analogue_article=anal["article"],
                    analogue_price=anal["price"]
                ) if results else sales_phrases["offer_only_analogue_12345678"].format(
                    model_year=model_year,
                    analogue_article=anal["article"],
                    analogue_price=anal["price"]
                )
            else:
                client_data["state"] = "done"  # Завершаем диалог после передачи
                lead_data = {
                    "requested_part": client_data["requested_part"],
                    "model_year": model_year,
                    "user_msg": user_msg
                }
                handover_to_manager(lead_data)
                results = retriever.search("handover", top_k=1)
                return results[0][0]["phrases"][0] if results else "К сожалению, деталь не найдена. Передам запрос менеджеру."

        return retriever.search("ask_model_year", top_k=1)[0][0]["phrases"][0]

    if client_data.get("state") == "offer_part":
        msg = user_msg.lower()
        orig = client_data.get("selected_original")
        anal = client_data.get("selected_analogue")

        if any(w in msg for w in ["оригинал", "давайте оригинал", "хочу оригинал"]):
            client_data["selected_part"] = orig
            client_data["state"] = "await_contact_info"
            return "Пожалуйста, укажите ваше имя и телефон для оформления счёта."

        if any(w in msg for w in ["аналог", "дешевле", "бюджетный"]):
            if anal:
                client_data["selected_part"] = anal
                client_data["state"] = "await_contact_info"
                return "Пожалуйста, укажите ваше имя и телефон для оформления счёта."
            return retriever.search("handover", top_k=1)[0][0]["phrases"][0]

        if any(w in msg for w in ["дорого", "цена"]):
            client_data["state"] = "handle_objection"
            results = retriever.search("objection_price", top_k=1)
            return results[0][0]["phrases"][0] if results else sales_phrases["objection_price_12345678"]

        if any(w in msg for w in ["аналог нормальный", "качество аналога"]):
            client_data["state"] = "handle_objection"
            results = retriever.search("objection_analogue_quality", top_k=1)
            return results[0][0]["phrases"][0] if results else "Аналог проверенного качества, часто используется в сервисах."

        if any(w in msg for w in ["да", "оформляйте", "заказать"]):
            client_data["selected_part"] = orig if orig else anal
            client_data["state"] = "await_contact_info"
            return "Пожалуйста, укажите ваше имя и телефон для оформления счёта."

        if any(w in msg for w in ["vin", "менеджера", "доставка", "гарантия"]):
            client_data["state"] = "done"  # Завершаем диалог
            lead_data = {
                "requested_part": client_data.get("requested_part", ""),
                "model_year": client_data.get("model_year", ""),
                "user_msg": user_msg
            }
            handover_to_manager(lead_data)
            results = retriever.search("handover", top_k=1)
            return results[0][0]["phrases"][0] if results else "Передам ваш вопрос менеджеру."

        return "Уточните, выбираем оригинал или аналог?"

    if client_data.get("state") == "handle_objection":
        msg = user_msg.lower()
        if any(w in msg for w in ["ок", "понятно", "давайте", "хорошо"]):
            client_data["selected_part"] = client_data.get("selected_original")
            client_data["state"] = "await_contact_info"
            return "Пожалуйста, укажите ваше имя и телефон для оформления счёта."

        if any(w in msg for w in ["аналог", "дешевле"]):
            if client_data.get("selected_analogue"):
                client_data["selected_part"] = client_data.get("selected_analogue")
                client_data["state"] = "await_contact_info"
                return "Пожалуйста, укажите ваше имя и телефон для оформления счёта."
            return retriever.search("handover", top_k=1)[0][0]["phrases"][0]

        return "Пожалуйста, выберите оригинал или аналог, чтобы оформить заказ."

    if client_data.get("state") == "await_contact_info":
        name_match = re.search(r"[А-Я][а-я]+", user_msg)
        phone_match = re.search(r"\+?\d{10,12}", user_msg)
        if name_match and phone_match:
            client_data["client_name"] = name_match.group()
            client_data["contact"] = phone_match.group()
            part = client_data.get("selected_part")
            if part:
                details = {
                    "client_name": client_data["client_name"],
                    "contact": client_data["contact"],
                    "part_article": part["article"],
                    "part_name": part["name"],
                    "price": part["price"],
                    "model_year": client_data.get("model_year")
                }
                send_invoice(details)
                client_data["state"] = "done"
                results = retriever.search("thanks", top_k=1)
                return results[0][0]["phrases"][0] if results else "Спасибо за заказ!"
        return "Пожалуйста, укажите имя и телефон, например: Иван, +79991234567."

    if client_data.get("state") == "done":
        return "Спасибо, что обратились! Всего доброго."

    return "Не понял ваш запрос, уточните, пожалуйста."

# Симуляция диалогов
def simulate_dialogs():
    dialogs = [
        {
            "requested_part": "моторчик омывателя",
            "messages": [
                "Здравствуйте!",
                "У меня XDrive 2009 года, хочу купить моторчик омывателя.",
                "Давайте оригинал.",
                "Иван, +79990000001"
            ]
        },
        {
            "requested_part": "задний фонарь",
            "messages": [
                "Добрый день.",
                "Cruiser 2012 года, интересует задний фонарь.",
                "Аналог нормальный?",
                "Да, оформляйте.",
                "Петр, +79990000002"
            ]
        },
        {
            "requested_part": "лямбда-зонд",
            "messages": [
                "Привет!",
                "Vento 2010, лямбда-зонд нужен.",
                "Слишком дорого, есть дешевле?",
                "Хорошо, давайте аналог.",
                "Анна, +79990000003"
            ]
        }
    ]

    retriever = Retriever()
    all_chunks = catalog_chunks + sales_templates
    retriever.build_index(all_chunks)

    for i, dialog in enumerate(dialogs, 1):
        print(f"\n--- Диалог {i} ---")
        client_data = {
            "state": "start",
            "requested_part": dialog["requested_part"],
            "client_name": f"Клиент {i}",
            "contact": f"+7999000000{i}"
        }
        history = []
        for msg in dialog["messages"]:
            print(f"Клиент: {msg}")
            agent_msg = agent_respond(history, msg, client_data, retriever)
            print(f"Агент: {agent_msg}")
            history.append((msg, agent_msg))

if __name__ == "__main__":
    simulate_dialogs()
