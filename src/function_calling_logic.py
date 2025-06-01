"""
Логика вызова функций взаимодействия с менеджером и отправки счета.
"""

from datetime import datetime
from typing import Dict, List

def send_invoice(details: Dict) -> Dict:
    """
    Отправка счета клиенту.
    details — словарь с информацией о заказе:
        {
            "client_name": str,
            "contact": str,
            "part_article": str,
            "part_name": str,
            "price": int,
            "model_year": str
        }
    Возвращает статус и детали.
    """
    required_fields = ["client_name", "contact", "part_article", "part_name", "price"]
    if not all(field in details for field in required_fields):
        return {"status": "error", "message": f"Missing required fields: {required_fields}"}

    total_price = details["price"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Отправляем счёт клиенту {details['client_name']} ({details['contact']}) "
          f"на деталь '{details['part_name']}' ({details['part_article']}) для {details.get('model_year', 'не указано')} "
          f"на сумму {total_price} ₽")
    return {
        "status": "invoice_sent",
        "details": {
            "client_name": details["client_name"],
            "contact": details["contact"],
            "items": [{
                "name": details["part_name"],
                "article": details["part_article"],
                "price": details["price"],
                "quantity": 1
            }],
            "total_price": total_price
        }
    }

def handover_to_manager(lead_data: Dict) -> Dict:
    """
    Передача лида живому менеджеру.
    lead_data — словарь с данными клиента и вопросами:
        {
            "requested_part": str (optional),
            "model_year": str (optional),
            "user_msg": str (optional, маппится в question),
            "client_name": str (optional),
            "contact": str (optional)
        }
    Возвращает статус и данные лида.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    question = lead_data.get("user_msg", "Не указан вопрос")
    context = (f"Запрошенная деталь: {lead_data.get('requested_part', 'не указана')}, "
               f"модель и год: {lead_data.get('model_year', 'не указаны')}")
    client_name = lead_data.get("client_name", "неизвестно")
    contact = lead_data.get("contact", "не указан")

    print(f"[{timestamp}] Передаём менеджеру лид от клиента {client_name} ({contact}) "
          f"с вопросом: '{question}' (контекст: {context})")
    return {
        "status": "lead_handed_over",
        "lead_data": {
            "client_name": client_name,
            "contact": contact,
            "question": question,
            "context": context
        }
    }

def should_send_invoice(dialog_state: Dict) -> bool:
    """
    Определяет, нужно ли отправить счёт.
    dialog_state — словарь с текущим состоянием диалога:
        {
            "state": str,
            "selected_part": Dict,
            "client_name": str (optional),
            "contact": str (optional)
        }
    Возвращает True, если клиент готов оформить заказ.
    """
    return (dialog_state.get("state") == "await_contact_info" and
            dialog_state.get("selected_part") and
            dialog_state.get("client_name") and
            dialog_state.get("contact"))

def should_handover_to_manager(dialog_state: Dict, user_msg: str = "") -> bool:
    """
    Определяет, когда нужно передать менеджеру.
    Условия:
    - Деталь не найдена (state=handover)
    - Запрос по VIN, доставке, гарантии
    - Клиент просит менеджера
    dialog_state содержит поля:
        {
            "state": str,
            "selected_part": Dict (optional),
            ...
        }
    user_msg — последнее сообщение клиента для проверки ключевых слов.
    """
    complex_keywords = ["vin", "доставка", "гарантия", "менеджер", "позвонить"]
    return (dialog_state.get("state") == "handover" or
            not dialog_state.get("selected_part") or
            any(keyword in user_msg.lower() for keyword in complex_keywords))
