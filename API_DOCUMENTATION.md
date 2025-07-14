# API Документация для Frontend-разработчика

## Обзор

Simulator_back - это FastAPI приложение для работы с гиперэластичными моделями материалов. Приложение предоставляет REST API для:
- Аутентификации пользователей с cookie-based сессиями
- Работы с анизотропными моделями (GOH, HOG)
- Работы с изотропными моделями (NeoHookean, MooneyRivlin, GeneralizedMooneyRivlin, Beda, Yeoh, Gent, Carroll)
- Загрузки данных, подгонки параметров и предсказаний
- Расчета энергии деформации

**Base URL:** `http://localhost:8000`
**Documentation:** `http://localhost:8000/swagger`

## Система аутентификации

### Cookie-based Authentication

Приложение использует сессионную аутентификацию с cookie. Все эндпоинты (кроме аутентификации) требуют наличия действующей сессии.

#### Настройки cookie:
- **Cookie Name:** `web-app-session-id`
- **TTL:** 14400 секунд (4 часа)
- **HttpOnly:** true
- **SameSite:** Lax
- **Secure:** false (для разработки)

#### Настройки CORS:
- **Allowed Origins:** `https://localhost:5173`
- **Credentials:** true
- **Methods:** `["*"]`
- **Headers:** `["*"]`

### Использование в frontend

#### Настройка fetch для работы с cookie:

```javascript
// Базовая конфигурация для всех запросов
const baseConfig = {
    credentials: 'include', // ВАЖНО: включает отправку cookie
    headers: {
        'Content-Type': 'application/json',
    }
};

// Функция для выполнения запросов
async function apiRequest(url, options = {}) {
    const config = {
        ...baseConfig,
        ...options,
        headers: {
            ...baseConfig.headers,
            ...options.headers,
        }
    };

    const response = await fetch(`http://localhost:8000${url}`, config);
    
    // Обработка ошибок аутентификации
    if (response.status === 401) {
        // Перенаправление на страницу авторизации
        window.location.href = '/login';
        throw new Error('Unauthorized');
    }
    
    return response;
}
```

#### Настройка axios для работы с cookie:

```javascript
import axios from 'axios';

// Создание экземпляра axios с базовыми настройками
const apiClient = axios.create({
    baseURL: 'http://localhost:8000',
    withCredentials: true, // ВАЖНО: включает отправку cookie
    timeout: 10000,
});

// Интерцептор для обработки ошибок аутентификации
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            // Перенаправление на страницу авторизации
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// Экспорт для использования в компонентах
export default apiClient;
```

### Endpoints аутентификации

#### POST /auth/login-cookie/

Вход в систему с получением cookie сессии.

**Request:**
```javascript
async function login(username, password) {
    const credentials = btoa(`${username}:${password}`);
    
    const response = await fetch('http://localhost:8000/auth/login-cookie/', {
        method: 'POST',
        headers: {
            'Authorization': `Basic ${credentials}`,
            'Content-Type': 'application/json',
        },
        credentials: 'include' // ВАЖНО: для получения cookie
    });
    
    if (!response.ok) {
        throw new Error('Login failed');
    }
    
    return await response.json();
}

// Пример использования
login('myusername', 'mypassword')
    .then(data => {
        console.log('Login successful:', data);
        // Cookie автоматически сохранится в браузере
    })
    .catch(error => {
        console.error('Login error:', error);
    });
```

**Response 200:**
```json
{
    "message": "Welcome, username"
}
```

#### GET /auth/logout-cookie/

Выход из системы с удалением cookie.

```javascript
async function logout() {
    const response = await fetch('http://localhost:8000/auth/logout-cookie/', {
        method: 'GET',
        credentials: 'include' // ВАЖНО: для отправки cookie
    });
    
    if (!response.ok) {
        throw new Error('Logout failed');
    }
    
    return await response.json();
}

// Пример использования
logout()
    .then(data => {
        console.log('Logout successful:', data);
        // Cookie автоматически удалится
    })
    .catch(error => {
        console.error('Logout error:', error);
    });
```

## Анизотропные модели

Базовый путь: `/modules/anisotropic`

### Поддерживаемые модели
- **GOH** - Gasser-Ogden-Holzapfel модель
- **HOG** - Holzapfel-Ogden-Gasser модель

### POST /modules/anisotropic/upload_model

Загрузка файлов данных и настройка модели для анизотропного анализа.

**Параметры:**
- `model_type` (обязательный): "GOH" или "HOG"
- `alpha` (опциональный): Угол α в радианах (0.0 ≤ α ≤ π)
- `kappa` (опциональный): Параметр κ (0.0 ≤ κ ≤ 1/3)
- `files` (обязательный): Массив CSV файлов

**Формат CSV файлов:**
```csv
lambda1,PE1,lambda2,PE2
1.0,0.0,1.0,0.0
1.1,0.5,1.0,0.0
1.2,1.2,1.0,0.0
1.3,2.1,1.0,0.0
```

**Пример использования:**

```javascript
async function uploadAnisotropicModel(files, modelType, alpha = null, kappa = null) {
    const formData = new FormData();
    
    // Добавляем тип модели
    formData.append('model_type', modelType);
    
    // Добавляем параметры, если они указаны
    if (alpha !== null) {
        formData.append('alpha', alpha.toString());
    }
    if (kappa !== null) {
        formData.append('kappa', kappa.toString());
    }
    
    // Добавляем файлы
    files.forEach(file => {
        formData.append('files', file);
    });
    
    const response = await fetch('http://localhost:8000/modules/anisotropic/upload_model', {
        method: 'POST',
        body: formData,
        credentials: 'include' // ВАЖНО: для отправки cookie
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
    }
    
    return await response.json();
}

// Пример использования с файлами из input
const fileInput = document.getElementById('fileInput');
const files = Array.from(fileInput.files);

uploadAnisotropicModel(files, 'GOH', 1.57, 0.2)
    .then(data => {
        console.log('Upload successful:', data);
    })
    .catch(error => {
        console.error('Upload error:', error);
    });
```

**React/TypeScript пример:**

```typescript
import React, { useState } from 'react';

interface UploadData {
    modelType: 'GOH' | 'HOG';
    alpha?: number;
    kappa?: number;
    files: File[];
}

const AnisotropicUpload: React.FC = () => {
    const [uploadData, setUploadData] = useState<UploadData>({
        modelType: 'GOH',
        files: []
    });

    const handleUpload = async () => {
        const formData = new FormData();
        
        formData.append('model_type', uploadData.modelType);
        
        if (uploadData.alpha !== undefined) {
            formData.append('alpha', uploadData.alpha.toString());
        }
        if (uploadData.kappa !== undefined) {
            formData.append('kappa', uploadData.kappa.toString());
        }
        
        uploadData.files.forEach(file => {
            formData.append('files', file);
        });
        
        try {
            const response = await fetch('http://localhost:8000/modules/anisotropic/upload_model', {
                method: 'POST',
                body: formData,
                credentials: 'include'
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail);
            }
            
            const result = await response.json();
            console.log('Upload successful:', result);
        } catch (error) {
            console.error('Upload error:', error);
        }
    };

    return (
        <div>
            <select 
                value={uploadData.modelType} 
                onChange={(e) => setUploadData({...uploadData, modelType: e.target.value as 'GOH' | 'HOG'})}
            >
                <option value="GOH">GOH</option>
                <option value="HOG">HOG</option>
            </select>
            
            <input 
                type="number" 
                step="0.01" 
                placeholder="Alpha (optional)"
                onChange={(e) => setUploadData({...uploadData, alpha: parseFloat(e.target.value) || undefined})}
            />
            
            <input 
                type="number" 
                step="0.01" 
                placeholder="Kappa (optional)"
                onChange={(e) => setUploadData({...uploadData, kappa: parseFloat(e.target.value) || undefined})}
            />
            
            <input 
                type="file" 
                multiple 
                accept=".csv"
                onChange={(e) => setUploadData({...uploadData, files: Array.from(e.target.files || [])})}
            />
            
            <button onClick={handleUpload}>Upload</button>
        </div>
    );
};
```

### POST /modules/anisotropic/fit

Подгонка параметров анизотропной модели.

```javascript
async function fitAnisotropicModel() {
    const response = await fetch('http://localhost:8000/modules/anisotropic/fit', {
        method: 'POST',
        credentials: 'include'
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Fit failed');
    }
    
    return await response.json();
}

// Пример использования
fitAnisotropicModel()
    .then(data => {
        console.log('Fit successful:', data);
        
        // Обработка параметров
        data.parameters.forEach(param => {
            console.log(`${param.name}: ${param.value}`);
        });
        
        // Обработка метрик
        data.metrics.forEach(metric => {
            console.log(`${metric.name}: ${metric.value}`);
        });
        
        // Обработка данных для графика
        console.log('Plot data:', data.plot_data);
    })
    .catch(error => {
        console.error('Fit error:', error);
    });
```

**Response 200:**
```json
{
    "status": "ok",
    "parameters": [
        {"name": "mu", "value": 0.5},
        {"name": "k1", "value": 0.3},
        {"name": "k2", "value": 0.1},
        {"name": "alpha", "value": 1.57},
        {"name": "kappa", "value": 0.2}
    ],
    "metrics": [
        {"name": "R²", "value": 0.95},
        {"name": "RMSE", "value": 0.05},
        {"name": "MAE", "value": 0.03}
    ],
    "plot_data": {
        "title": "Model Fitting Results",
        "x_label": "Stretch",
        "y_label": "Stress (MPa)",
        "lines": [
            {
                "name": "Experimental",
                "x": [1.0, 1.1, 1.2, 1.3],
                "y": [0.0, 0.5, 1.2, 2.1]
            },
            {
                "name": "Fitted",
                "x": [1.0, 1.1, 1.2, 1.3],
                "y": [0.0, 0.48, 1.18, 2.08]
            }
        ]
    }
}
```

### POST /modules/anisotropic/predict

Предсказания с использованием подогнанной модели.

```javascript
async function predictAnisotropic(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/modules/anisotropic/predict', {
        method: 'POST',
        body: formData,
        credentials: 'include'
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
    }
    
    return await response.json();
}

// Пример использования
const predictionFile = document.getElementById('predictionFileInput').files[0];
predictAnisotropic(predictionFile)
    .then(data => {
        console.log('Prediction successful:', data);
        
        // Обработка метрик
        data.metrics.forEach(metric => {
            console.log(`${metric.name}: ${metric.value}`);
        });
        
        // Построение графика
        plotResults(data.plot_data);
    })
    .catch(error => {
        console.error('Prediction error:', error);
    });
```

### DELETE /modules/anisotropic/file/{filename}

Удаление конкретного файла.

```javascript
async function deleteAnisotropicFile(filename) {
    const response = await fetch(`http://localhost:8000/modules/anisotropic/file/${filename}`, {
        method: 'DELETE',
        credentials: 'include'
    });
    
    if (!response.ok) {
        if (response.status === 404) {
            throw new Error('File not found');
        }
        throw new Error('Delete failed');
    }
    
    // Успешное удаление возвращает 204 No Content
    return true;
}
```

### DELETE /modules/anisotropic/clear_data

Очистка всех данных сессии.

```javascript
async function clearAnisotropicData() {
    const response = await fetch('http://localhost:8000/modules/anisotropic/clear_data', {
        method: 'DELETE',
        credentials: 'include'
    });
    
    if (!response.ok) {
        throw new Error('Clear failed');
    }
    
    return true;
}
```

## Изотропные модели

Базовый путь: `/modules/isotropic`

### Поддерживаемые модели
- **NeoHookean** - Неохуковская модель
- **MooneyRivlin** - Модель Муни-Ривлина
- **GeneralizedMooneyRivlin** - Обобщенная модель Муни-Ривлина
- **Beda** - Модель Беда
- **Yeoh** - Модель Йео
- **Gent** - Модель Джента
- **Carroll** - Модель Кэрролла

### POST /modules/isotropic/upload_model

Загрузка файлов данных для изотропного анализа.

**Параметры:**
- `hyperlastic_model` (обязательный): Одна из поддерживаемых моделей
- `files` (обязательный): Массив файлов (.csv, .xls, .xlsx)

**Формат CSV файлов:**
```csv
lambda_x,lambda_y,stress_x_mpa,stress_y_mpa
1.0,1.0,0.0,0.0
1.1,0.95,0.5,0.0
1.2,0.91,1.2,0.0
1.3,0.87,2.1,0.0
```

**Пример использования:**

```javascript
async function uploadIsotropicModel(files, modelType) {
    const formData = new FormData();
    
    formData.append('hyperlastic_model', modelType);
    
    files.forEach(file => {
        formData.append('files', file);
    });
    
    const response = await fetch('http://localhost:8000/modules/isotropic/upload_model', {
        method: 'POST',
        body: formData,
        credentials: 'include'
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
    }
    
    return await response.json();
}

// Пример использования
const files = Array.from(document.getElementById('fileInput').files);
uploadIsotropicModel(files, 'NeoHookean')
    .then(data => {
        console.log('Upload successful:', data);
    })
    .catch(error => {
        console.error('Upload error:', error);
    });
```

**React компонент для выбора модели:**

```typescript
import React, { useState } from 'react';

const IsotropicModelSelector: React.FC = () => {
    const [selectedModel, setSelectedModel] = useState<string>('NeoHookean');
    const [files, setFiles] = useState<File[]>([]);

    const modelTypes = [
        'NeoHookean',
        'MooneyRivlin',
        'GeneralizedMooneyRivlin',
        'Beda',
        'Yeoh',
        'Gent',
        'Carroll'
    ];

    const handleUpload = async () => {
        const formData = new FormData();
        formData.append('hyperlastic_model', selectedModel);
        
        files.forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch('http://localhost:8000/modules/isotropic/upload_model', {
                method: 'POST',
                body: formData,
                credentials: 'include'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail);
            }

            const result = await response.json();
            console.log('Upload successful:', result);
        } catch (error) {
            console.error('Upload error:', error);
        }
    };

    return (
        <div>
            <select 
                value={selectedModel} 
                onChange={(e) => setSelectedModel(e.target.value)}
            >
                {modelTypes.map(model => (
                    <option key={model} value={model}>{model}</option>
                ))}
            </select>
            
            <input 
                type="file" 
                multiple 
                accept=".csv,.xls,.xlsx"
                onChange={(e) => setFiles(Array.from(e.target.files || []))}
            />
            
            <button onClick={handleUpload}>Upload</button>
        </div>
    );
};
```

### POST /modules/isotropic/fit

Подгонка параметров изотропной модели.

```javascript
async function fitIsotropicModel() {
    const response = await fetch('http://localhost:8000/modules/isotropic/fit', {
        method: 'POST',
        credentials: 'include'
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Fit failed');
    }
    
    return await response.json();
}

// Пример использования
fitIsotropicModel()
    .then(data => {
        console.log('Fit successful:', data);
        
        // Обработка параметров
        data.parameters.forEach(param => {
            console.log(`${param.name}: ${param.value}`);
        });
        
        // Обработка метрик
        data.metrics.forEach(metric => {
            console.log(`${metric.name}: ${metric.value}`);
        });
        
        // Построение графика
        plotResults(data.plot_data);
    })
    .catch(error => {
        console.error('Fit error:', error);
    });
```

### POST /modules/isotropic/predict

Предсказания с использованием подогнанной изотропной модели.

```javascript
async function predictIsotropic(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/modules/isotropic/predict', {
        method: 'POST',
        body: formData,
        credentials: 'include'
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
    }
    
    return await response.json();
}
```

### GET /modules/isotropic/calculate_energy

Получение файла энергии деформации.

```javascript
async function calculateEnergy() {
    const response = await fetch('http://localhost:8000/modules/isotropic/calculate_energy', {
        method: 'GET',
        credentials: 'include'
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Energy calculation failed');
    }
    
    // Ответ возвращается как plain text
    return await response.text();
}

// Пример использования
calculateEnergy()
    .then(energyText => {
        console.log('Energy file content:', energyText);
        
        // Можно сохранить в файл
        const blob = new Blob([energyText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'energy.energy';
        a.click();
        URL.revokeObjectURL(url);
    })
    .catch(error => {
        console.error('Energy calculation error:', error);
    });
```

### DELETE /modules/isotropic/file/{filename}

Удаление конкретного файла.

```javascript
async function deleteIsotropicFile(filename) {
    const response = await fetch(`http://localhost:8000/modules/isotropic/file/${filename}`, {
        method: 'DELETE',
        credentials: 'include'
    });
    
    if (!response.ok) {
        if (response.status === 404) {
            throw new Error('File not found');
        }
        throw new Error('Delete failed');
    }
    
    return true;
}
```

### DELETE /modules/isotropic/clear_data

Очистка всех данных сессии.

```javascript
async function clearIsotropicData() {
    const response = await fetch('http://localhost:8000/modules/isotropic/clear_data', {
        method: 'DELETE',
        credentials: 'include'
    });
    
    if (!response.ok) {
        throw new Error('Clear failed');
    }
    
    return true;
}
```

## Построение графиков

### Использование Chart.js

```javascript
import Chart from 'chart.js/auto';

function plotResults(plotData) {
    const ctx = document.getElementById('myChart').getContext('2d');
    
    const datasets = plotData.lines.map((line, index) => ({
        label: line.name,
        data: line.x.map((x, i) => ({ x: x, y: line.y[i] })),
        borderColor: `hsl(${index * 60}, 70%, 50%)`,
        backgroundColor: `hsla(${index * 60}, 70%, 50%, 0.1)`,
        fill: false,
        tension: 0.1
    }));

    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: plotData.title || plotData.name
                },
                legend: {
                    display: true
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    display: true,
                    title: {
                        display: true,
                        text: plotData.x_label
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: plotData.y_label
                    }
                }
            }
        }
    });
}
```

### Использование Plotly.js

```javascript
import Plotly from 'plotly.js-dist';

function plotResultsPlotly(plotData) {
    const traces = plotData.lines.map(line => ({
        x: line.x,
        y: line.y,
        type: 'scatter',
        mode: 'lines+markers',
        name: line.name
    }));

    const layout = {
        title: plotData.title || plotData.name,
        xaxis: {
            title: plotData.x_label
        },
        yaxis: {
            title: plotData.y_label
        }
    };

    Plotly.newPlot('plotDiv', traces, layout);
}
```

## Полные примеры использования

### Полный цикл анизотропной модели

```javascript
class AnisotropicModelProcessor {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
    }

    async login(username, password) {
        const credentials = btoa(`${username}:${password}`);
        
        const response = await fetch(`${this.baseUrl}/auth/login-cookie/`, {
            method: 'POST',
            headers: {
                'Authorization': `Basic ${credentials}`
            },
            credentials: 'include'
        });

        if (!response.ok) {
            throw new Error('Login failed');
        }

        return await response.json();
    }

    async uploadModel(files, modelType, alpha = null, kappa = null) {
        const formData = new FormData();
        
        formData.append('model_type', modelType);
        
        if (alpha !== null) {
            formData.append('alpha', alpha.toString());
        }
        if (kappa !== null) {
            formData.append('kappa', kappa.toString());
        }
        
        files.forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch(`${this.baseUrl}/modules/anisotropic/upload_model`, {
            method: 'POST',
            body: formData,
            credentials: 'include'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail);
        }

        return await response.json();
    }

    async fitModel() {
        const response = await fetch(`${this.baseUrl}/modules/anisotropic/fit`, {
            method: 'POST',
            credentials: 'include'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail);
        }

        return await response.json();
    }

    async predict(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseUrl}/modules/anisotropic/predict`, {
            method: 'POST',
            body: formData,
            credentials: 'include'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail);
        }

        return await response.json();
    }

    async processFullWorkflow(username, password, files, modelType, predictionFile, alpha = null, kappa = null) {
        try {
            // 1. Авторизация
            console.log('Logging in...');
            await this.login(username, password);

            // 2. Загрузка модели
            console.log('Uploading model...');
            const uploadResult = await this.uploadModel(files, modelType, alpha, kappa);
            console.log('Upload result:', uploadResult);

            // 3. Подгонка модели
            console.log('Fitting model...');
            const fitResult = await this.fitModel();
            console.log('Fit result:', fitResult);

            // 4. Предсказание
            console.log('Making predictions...');
            const predictionResult = await this.predict(predictionFile);
            console.log('Prediction result:', predictionResult);

            return {
                upload: uploadResult,
                fit: fitResult,
                prediction: predictionResult
            };

        } catch (error) {
            console.error('Workflow error:', error);
            throw error;
        }
    }
}

// Использование
const processor = new AnisotropicModelProcessor();

// Получение файлов из input
const trainingFiles = Array.from(document.getElementById('trainingFiles').files);
const predictionFile = document.getElementById('predictionFile').files[0];

processor.processFullWorkflow(
    'username', 
    'password', 
    trainingFiles, 
    'GOH', 
    predictionFile,
    1.57, // alpha
    0.2   // kappa
).then(results => {
    console.log('Full workflow completed:', results);
    
    // Построение графиков
    plotResults(results.fit.plot_data);
    plotResults(results.prediction.plot_data);
    
}).catch(error => {
    console.error('Workflow failed:', error);
});
```

### React Hook для работы с API

```typescript
import { useState, useCallback } from 'react';

interface ApiError {
    detail: string;
}

interface UseApiResult<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
    execute: (...args: any[]) => Promise<T>;
}

export function useApi<T>(
    apiFunction: (...args: any[]) => Promise<T>
): UseApiResult<T> {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const execute = useCallback(async (...args: any[]) => {
        setLoading(true);
        setError(null);

        try {
            const result = await apiFunction(...args);
            setData(result);
            return result;
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Unknown error';
            setError(errorMessage);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [apiFunction]);

    return { data, loading, error, execute };
}

// Использование в компоненте
const MyComponent: React.FC = () => {
    const { data, loading, error, execute } = useApi(fitAnisotropicModel);

    const handleFit = async () => {
        try {
            const result = await execute();
            console.log('Fit successful:', result);
        } catch (error) {
            console.error('Fit failed:', error);
        }
    };

    return (
        <div>
            <button onClick={handleFit} disabled={loading}>
                {loading ? 'Fitting...' : 'Fit Model'}
            </button>
            {error && <div className="error">{error}</div>}
            {data && <div>Fit completed successfully!</div>}
        </div>
    );
};
```

## Обработка ошибок

### Централизованная обработка ошибок

```javascript
class ApiErrorHandler {
    static handle(error, context = '') {
        if (error.response) {
            // Ошибка с ответом от сервера
            const status = error.response.status;
            const message = error.response.data?.detail || error.message;

            switch (status) {
                case 400:
                    console.error(`Bad Request in ${context}:`, message);
                    return `Неверные данные: ${message}`;
                
                case 401:
                    console.error(`Unauthorized in ${context}:`, message);
                    // Перенаправление на авторизацию
                    window.location.href = '/login';
                    return 'Требуется авторизация';
                
                case 404:
                    console.error(`Not Found in ${context}:`, message);
                    return `Ресурс не найден: ${message}`;
                
                case 500:
                    console.error(`Server Error in ${context}:`, message);
                    return `Ошибка сервера: ${message}`;
                
                default:
                    console.error(`Unknown Error in ${context}:`, message);
                    return `Неизвестная ошибка: ${message}`;
            }
        } else if (error.request) {
            // Ошибка сети
            console.error(`Network Error in ${context}:`, error.message);
            return 'Ошибка сети. Проверьте подключение к интернету.';
        } else {
            // Другие ошибки
            console.error(`Error in ${context}:`, error.message);
            return error.message;
        }
    }
}

// Использование
try {
    const result = await fitAnisotropicModel();
} catch (error) {
    const errorMessage = ApiErrorHandler.handle(error, 'Fit Model');
    // Показать ошибку пользователю
    showErrorToUser(errorMessage);
}
```

## Рекомендации по безопасности

### 1. Защита от CSRF

```javascript
// Получение CSRF токена (если используется)
async function getCsrfToken() {
    const response = await fetch('http://localhost:8000/csrf-token', {
        credentials: 'include'
    });
    const data = await response.json();
    return data.csrf_token;
}

// Использование в запросах
const csrfToken = await getCsrfToken();
const response = await fetch('http://localhost:8000/modules/anisotropic/fit', {
    method: 'POST',
    headers: {
        'X-CSRFToken': csrfToken
    },
    credentials: 'include'
});
```

### 2. Валидация данных на frontend

```javascript
function validateFile(file) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];
    
    if (file.size > maxSize) {
        throw new Error('Файл слишком большой. Максимальный размер: 10MB');
    }
    
    if (!allowedTypes.includes(file.type)) {
        throw new Error('Неподдерживаемый тип файла. Используйте CSV или Excel файлы.');
    }
    
    return true;
}
```

### 3. Таймауты для запросов

```javascript
const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 секунд

try {
    const response = await fetch('http://localhost:8000/modules/anisotropic/fit', {
        method: 'POST',
        credentials: 'include',
        signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
        throw new Error('Request failed');
    }
    
    return await response.json();
} catch (error) {
    if (error.name === 'AbortError') {
        throw new Error('Запрос превысил лимит времени');
    }
    throw error;
}
```

## Тестирование API

### Unit тесты с Jest

```javascript
// api.test.js
import { jest } from '@jest/globals';
import { fitAnisotropicModel } from './api';

// Мокируем fetch
global.fetch = jest.fn();

describe('API Tests', () => {
    beforeEach(() => {
        fetch.mockClear();
    });

    test('fitAnisotropicModel success', async () => {
        const mockResponse = {
            status: 'ok',
            parameters: [{ name: 'mu', value: 0.5 }],
            metrics: [{ name: 'R²', value: 0.95 }],
            plot_data: { title: 'Test Plot', lines: [] }
        };

        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => mockResponse
        });

        const result = await fitAnisotropicModel();
        
        expect(fetch).toHaveBeenCalledWith(
            'http://localhost:8000/modules/anisotropic/fit',
            expect.objectContaining({
                method: 'POST',
                credentials: 'include'
            })
        );
        
        expect(result).toEqual(mockResponse);
    });

    test('fitAnisotropicModel error', async () => {
        fetch.mockResolvedValueOnce({
            ok: false,
            status: 400,
            json: async () => ({ detail: 'Bad request' })
        });

        await expect(fitAnisotropicModel()).rejects.toThrow('Bad request');
    });
});
```

### Integration тесты с Cypress

```javascript
// cypress/integration/api.spec.js
describe('API Integration Tests', () => {
    beforeEach(() => {
        // Логин перед каждым тестом
        cy.login('testuser', 'testpass');
    });

    it('should upload and fit anisotropic model', () => {
        // Загрузка файла
        cy.fixture('test-data.csv').then((csvContent) => {
            cy.get('[data-cy=file-input]').attachFile({
                fileContent: csvContent,
                fileName: 'test-data.csv',
                mimeType: 'text/csv'
            });
        });

        // Выбор модели
        cy.get('[data-cy=model-select]').select('GOH');

        // Загрузка
        cy.get('[data-cy=upload-button]').click();
        cy.get('[data-cy=upload-success]').should('be.visible');

        // Подгонка модели
        cy.get('[data-cy=fit-button]').click();
        cy.get('[data-cy=fit-success]').should('be.visible');

        // Проверка результатов
        cy.get('[data-cy=parameters-table]').should('contain', 'mu');
        cy.get('[data-cy=metrics-table]').should('contain', 'R²');
    });
});
```

## Заключение

Данная документация предоставляет полное руководство по использованию API Simulator_back для frontend разработчиков. Основные моменты:

1. **Обязательно используйте** `credentials: 'include'` во всех запросах
2. **Настройте CORS** корректно для вашего домена
3. **Обрабатывайте ошибки** 401 для перенаправления на авторизацию
4. **Валидируйте данные** перед отправкой
5. **Используйте таймауты** для долгих операций
6. **Тестируйте API** функциональность

Все endpoints требуют авторизации через cookie-сессию, кроме endpoint'ов авторизации. Для получения актуальной информации о параметрах и форматах данных используйте Swagger документацию по адресу `/swagger`.

При возникновении проблем проверьте:
- Корректность настроек CORS
- Валидность сессии
- Формат отправляемых данных
- Статус коды ответов

Удачной разработки! 