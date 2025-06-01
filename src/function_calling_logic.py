from datetime import datetime

def send_invoice(data: dict):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Отправляем счёт клиенту {data['client_name']} ({data['contact']}) на деталь '{data['part_name']}' ({data['part_article']}) для {data['model_year']} на сумму {data['price']} ₽")

def handover_to_manager(data: dict):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Передаём лид менеджеру: запрошена деталь '{data['requested_part']}' для {data['model_year']}. Сообщение клиента: '{data['user_msg']}'")
