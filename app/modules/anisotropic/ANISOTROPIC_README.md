# Anisotropic Hyperelastic Material Modeling Module

## Обзор

Модуль `anisotropic` представляет собой полную переработку консольного приложения `HOG_fit2.py` в веб-модуль с соблюдением принципов чистого кода и SOLID.

## Архитектура

Модуль построен по многослойной архитектуре:

### Слои архитектуры:

1. **Handlers** (`handlers.py`) - API endpoints для работы с анизотропными моделями
2. **Service** (`server.py`) - Бизнес-логика и оркестрация
3. **Cache** (`anisotropic_cache.py`) - Слой кеширования в Redis
4. **Solver** (`solver/`) - Алгоритмы и математические модели
5. **Dependency Injection** (`anisotropic_dependency.py`) - Внедрение зависимостей

### Solver подмодули:

- `config.py` - Конфигурация и константы
- `models.py` - Реализации моделей GOH и HOG
- `parameter_optimizer.py` - Оптимизация параметров
- `evaluator.py` - Оценка качества модели
- `anisotropic_solver.py` - Основной решатель

## Поддерживаемые модели

### 1. GOH (Gasser-Ogden-Holzapfel) Model
- Модель для анизотропных гиперупругих материалов
- Учитывает направление волокон и их дисперсию

### 2. HOG (Holzapfel-Ogden-Gasser) Model  
- Альтернативная формулировка анизотропной модели
- Различная функция энергии деформации

## API Endpoints

### `/modules/anisotropic/upload_model`
**POST** - Загрузка данных и конфигурация модели

**Параметры:**
- `model_type`: "GOH" или "HOG"
- `kappa`: Фиксированное значение параметра κ (опционально)
- `alpha`: Фиксированный угол α в радианах (опционально) 
- `files`: CSV файлы с данными

**Формат данных:**
```csv
lambda1,PE1,lambda2,PE2
1.1,0.5,1.0,0.0
1.2,1.2,1.0,0.0
...
```

### `/modules/anisotropic/fit`
**POST** - Подгонка параметров модели

**Возвращает:**
- Оптимизированные параметры (μ, k₁, k₂, κ, α)
- Метрики качества (R², RMSE, MAE)
- Данные для графиков
- Информацию о сходимости

### `/modules/anisotropic/predict`
**POST** - Предсказания на новых данных

**Параметры:**
- `file`: CSV файл с данными для предсказания

### `/modules/anisotropic/session_info`
**GET** - Информация о текущей сессии

### `/modules/anisotropic/clear_data`
**DELETE** - Очистка всех данных сессии

## Принципы чистого кода

### SOLID принципы:

1. **Single Responsibility** - Каждый класс имеет одну ответственность
   - `ModelEvaluator` - только оценка качества
   - `ParameterOptimizer` - только оптимизация
   - `AnisotropicCache` - только кеширование

2. **Open/Closed** - Легко добавлять новые модели через `ModelFactory`

3. **Liskov Substitution** - Все модели наследуют `AnisotropicModel`

4. **Interface Segregation** - Интерфейсы разделены по функциональности

5. **Dependency Inversion** - Зависимости инжектируются через FastAPI DI

### Улучшения по сравнению с оригинальным кодом:

#### ❌ Проблемы в `HOG_fit2.py`:
- Глобальные переменные
- Монолитные функции (>100 строк)
- Смешанная логика UI и бизнес-логики
- Отсутствие обработки ошибок
- Дублирование кода
- Отсутствие типизации
- Сложность тестирования

#### ✅ Решения в новом модуле:
- Инкапсуляция данных в классы
- Малые функции с единственной ответственностью
- Разделение UI (API) и бизнес-логики
- Полная обработка ошибок с логированием
- DRY принцип - переиспользование кода
- Полная типизация с mypy
- Модульность для легкого тестирования

## Пример использования

### 1. Загрузка данных
```python
import requests

# Загрузка CSV файла с данными
files = {'files': open('data.csv', 'rb')}
data = {
    'model_type': 'GOH',
    'kappa': 0.1,  # Опционально
    'alpha': 1.57  # Опционально (π/2)
}

response = requests.post('/modules/anisotropic/upload_model', 
                        files=files, data=data)
```

### 2. Подгонка модели
```python
response = requests.post('/modules/anisotropic/fit')
result = response.json()

print(f"Model: {result['model_type']}")
print(f"R² P11: {result['metrics'][0]['value']:.4f}")
print(f"R² P22: {result['metrics'][1]['value']:.4f}")
```

### 3. Предсказания
```python
files = {'file': open('test_data.csv', 'rb')}
response = requests.post('/modules/anisotropic/predict', files=files)
predictions = response.json()
```

## Расширяемость

### Добавление новой модели:

1. Создать класс наследующий `AnisotropicModel`:
```python
class NewModel(AnisotropicModel):
    def compute_stress(self, lam1, lam2, params):
        # Реализация новой модели
        pass
```

2. Зарегистрировать в `ModelFactory`:
```python
_models = {
    AnisotropicModelType.GOH: GOHModel,
    AnisotropicModelType.HOG: HOGModel,
    AnisotropicModelType.NEW: NewModel,  # Новая модель
}
```

## Производительность

- Векторизованные вычисления с NumPy
- Кеширование в Redis для сессий
- Асинхронная обработка запросов
- Оптимизированные алгоритмы с autograd

## Безопасность

- Валидация всех входных данных
- Изоляция сессий пользователей
- Безопасная обработка файлов
- Логирование всех операций

## Заключение

Новый анизотропный модуль представляет собой современное, масштабируемое и поддерживаемое решение для веб-приложения, полностью заменяющее консольный скрипт и следующее лучшим практикам разработки ПО. 