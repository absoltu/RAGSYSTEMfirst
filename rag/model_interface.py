import requests
from typing import Optional, Dict, Any
import json

from config import (
    LM_STUDIO_BASE_URL,
    CHAT_MODEL,
    SYSTEM_PROMPT,
    TIMEOUT_REQUEST
)


class ModelInterface:
    """
    Интерфейс для общения с языковой моделью через LM Studio API.
    """

    def __init__(self, base_url: str = LM_STUDIO_BASE_URL, model: str = CHAT_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.chat_url = f"{self.base_url}/chat/completions"

    def generate_response(
        self,
        context: str,
        query: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Генерирует ответ на основе контекста и запроса.

        :param context: Контекст из поиска
        :param query: Запрос пользователя
        :param max_tokens: Максимальное количество токенов в ответе
        :param temperature: Температура генерации
        :param system_prompt: Системный промпт (если None, используется дефолтный)
        :return: Сгенерированный ответ
        """
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        # Формируем промпт
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Подготавливаем сообщения для chat API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Подготавливаем payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        try:
            # Отправляем запрос
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT_REQUEST
            )

            response.raise_for_status()

            result = response.json()

            # Извлекаем ответ
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return "No response generated."

        except requests.exceptions.RequestException as e:
            return f"Error communicating with model: {str(e)}"
        except json.JSONDecodeError as e:
            return f"Error parsing model response: {str(e)}"
        except KeyError as e:
            return f"Unexpected response format: {str(e)}"

    def check_connection(self) -> bool:
        """
        Проверяет соединение с моделью.

        :return: True если соединение работает
        """
        try:
            # Простой запрос для проверки
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1
            }

            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            return response.status_code == 200

        except:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получает информацию о модели.

        :return: Информация о модели
        """
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return response.json()
        except:
            return {"error": "Could not retrieve model information"}