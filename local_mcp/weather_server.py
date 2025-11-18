#!/usr/bin/env python3
"""
Локальный MCP сервер для получения погоды через API Яндекс.Погоды.
Запускается на порту 8080 и принимает запросы с координатами.
"""

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import requests
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Порт сервера
PORT = 8080

# URL API Яндекс.Погоды
YANDEX_WEATHER_API_URL = "https://api.weather.yandex.ru/v2/forecast"


class WeatherRequestHandler(BaseHTTPRequestHandler):
    """Обработчик HTTP запросов для получения погоды."""
    
    def do_GET(self):
        """Обработка GET запросов."""
        try:
            # Парсинг URL и параметров
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            
            logger.info(f"Входящий GET запрос: {self.path}")
            logger.info(f"Параметры запроса: {query_params}")
            
            # Извлечение координат
            latitude = query_params.get('latitude', [None])[0]
            longitude = query_params.get('longitude', [None])[0]
            
            # Обработка запроса
            response_data = self._handle_weather_request(latitude, longitude)
            
            # Отправка ответа
            self._send_json_response(response_data, 200 if 'error' not in response_data else 400)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке GET запроса: {e}", exc_info=True)
            error_response = {"error": f"Внутренняя ошибка сервера: {str(e)}"}
            self._send_json_response(error_response, 500)
    
    def do_POST(self):
        """Обработка POST запросов."""
        try:
            # Чтение тела запроса
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            
            logger.info(f"Входящий POST запрос: {self.path}")
            logger.info(f"Тело запроса: {body}")
            
            # Парсинг JSON
            try:
                request_data = json.loads(body) if body else {}
            except json.JSONDecodeError as e:
                error_response = {"error": f"Некорректный JSON: {str(e)}"}
                self._send_json_response(error_response, 400)
                return
            
            # Извлечение координат
            latitude = request_data.get('latitude')
            longitude = request_data.get('longitude')
            
            # Обработка запроса
            response_data = self._handle_weather_request(latitude, longitude)
            
            # Отправка ответа
            self._send_json_response(response_data, 200 if 'error' not in response_data else 400)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке POST запроса: {e}", exc_info=True)
            error_response = {"error": f"Внутренняя ошибка сервера: {str(e)}"}
            self._send_json_response(error_response, 500)
    
    def _handle_weather_request(self, latitude, longitude):
        """
        Обрабатывает запрос на получение погоды.
        
        Args:
            latitude: Широта
            longitude: Долгота
            
        Returns:
            dict: JSON с данными о погоде или ошибкой
        """
        # Проверка наличия координат
        if latitude is None or longitude is None:
            logger.warning("Отсутствуют координаты в запросе")
            return {"error": "Необходимо указать latitude и longitude"}
        
        # Проверка и преобразование координат
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            logger.warning(f"Некорректные координаты: lat={latitude}, lon={longitude}")
            return {"error": "Координаты должны быть числами"}
        
        # Проверка диапазона координат
        if not (-90 <= latitude <= 90):
            logger.warning(f"Широта вне допустимого диапазона: {latitude}")
            return {"error": "Широта должна быть в диапазоне от -90 до 90"}
        
        if not (-180 <= longitude <= 180):
            logger.warning(f"Долгота вне допустимого диапазона: {longitude}")
            return {"error": "Долгота должна быть в диапазоне от -180 до 180"}
        
        # Получение API ключа
        api_key = os.getenv('YANDEX_WEATHER_API_KEY')
        if not api_key:
            logger.error("Переменная окружения YANDEX_WEATHER_API_KEY не установлена")
            return {"error": "API ключ не настроен. Установите переменную окружения YANDEX_WEATHER_API_KEY"}
        
        # Проверка, что ключ не пустой
        api_key = api_key.strip()
        if not api_key:
            logger.error("API ключ пустой")
            return {"error": "API ключ пустой. Проверьте переменную окружения YANDEX_WEATHER_API_KEY в .env файле"}
        
        # Запрос к API Яндекс.Погоды
        try:
            logger.info(f"Запрос к API Яндекс.Погоды для координат: lat={latitude}, lon={longitude}")
            
            headers = {
                'X-Yandex-Weather-Key': api_key
            }
            
            params = {
                'lat': latitude,
                'lon': longitude
            }
            
            response = requests.get(YANDEX_WEATHER_API_URL, headers=headers, params=params, timeout=10)
            
            logger.info(f"Ответ от API Яндекс.Погоды: статус {response.status_code}")
            
            # Проверка статуса ответа
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Ошибка API Яндекс.Погоды: статус {response.status_code}, ответ: {error_text}")
                
                # Более детальное сообщение об ошибке
                if response.status_code == 403:
                    return {
                        "error": "Ошибка доступа (403). Проверьте правильность API ключа YANDEX_WEATHER_API_KEY",
                        "details": error_text if error_text else "Forbidden"
                    }
                elif response.status_code == 401:
                    return {
                        "error": "Ошибка аутентификации (401). Неверный API ключ",
                        "details": error_text if error_text else "Unauthorized"
                    }
                else:
                    return {
                        "error": f"Ошибка API Яндекс.Погоды: статус {response.status_code}",
                        "details": error_text if error_text else "Unknown error"
                    }
            
            # Парсинг ответа
            weather_data = response.json()
            logger.info(f"Полный ответ от API: {json.dumps(weather_data, ensure_ascii=False, indent=2)}")
            
            # Извлечение нужных данных
            try:
                # Для /v2/forecast структура: weather_data['fact']
                fact = weather_data.get('fact', {})
                
                if not fact:
                    # Попробуем альтернативную структуру
                    logger.warning("Поле 'fact' не найдено, проверяю альтернативную структуру")
                    fact = weather_data.get('forecasts', [{}])[0].get('parts', {}).get('day', {}) if weather_data.get('forecasts') else {}
                
                result = {
                    "temperature": fact.get('temp'),
                    "condition": fact.get('condition'),
                    "wind_speed": fact.get('wind_speed'),
                    "humidity": fact.get('humidity')
                }
                
                # Проверка, что хотя бы некоторые данные получены
                if all(v is None for v in result.values()):
                    logger.error(f"Не удалось извлечь данные из ответа. Структура: {weather_data}")
                    return {"error": "Не удалось извлечь данные о погоде из ответа API", "raw_response": weather_data}
                
                logger.info(f"Успешный ответ: {result}")
                return result
                
            except (KeyError, AttributeError, IndexError) as e:
                logger.error(f"Ошибка при парсинге ответа API: {e}, данные: {weather_data}")
                return {"error": f"Неожиданный формат ответа от API: {str(e)}", "raw_response": weather_data}
            
        except requests.exceptions.Timeout:
            logger.error("Таймаут при запросе к API Яндекс.Погоды")
            return {"error": "Таймаут при запросе к API Яндекс.Погоды"}
        
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ошибка соединения с API Яндекс.Погоды: {e}")
            return {"error": f"Ошибка соединения с API: {str(e)}"}
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к API Яндекс.Погоды: {e}")
            return {"error": f"Ошибка сети: {str(e)}"}
        
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
            return {"error": f"Неожиданная ошибка: {str(e)}"}
    
    def _send_json_response(self, data, status_code=200):
        """
        Отправляет JSON ответ клиенту.
        
        Args:
            data: Данные для отправки (будут преобразованы в JSON)
            status_code: HTTP статус код
        """
        json_data = json.dumps(data, ensure_ascii=False, indent=2)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json_data.encode('utf-8'))
        
        logger.info(f"Отправлен ответ со статусом {status_code}: {json_data}")
    
    def log_message(self, format, *args):
        """Переопределение метода логирования для использования нашего logger."""
        logger.info(f"{self.address_string()} - {format % args}")


def run_server(port=PORT):
    """Запускает HTTP сервер."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, WeatherRequestHandler)
    
    logger.info(f"Запуск MCP сервера погоды на порту {port}")
    logger.info(f"Сервер доступен по адресу: http://localhost:{port}")
    logger.info("Для остановки нажмите Ctrl+C")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
        httpd.shutdown()
        logger.info("Сервер остановлен")


if __name__ == '__main__':
    run_server()

