@echo off
setlocal

REM Переходим в папку, где лежит bat
cd /d %~dp0

echo.
echo === neweds: fMRI Connectivity Analysis ===
echo.

REM Проверка Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден. Установи Python 3.10+ и добавь в PATH.
    pause
    exit /b 1
)

REM Проверяем streamlit
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Установка зависимостей...
    python -m pip install --upgrade pip wheel
    python -m pip install "setuptools<81"
    python -m pip install --upgrade --prefer-binary -r requirements.txt
)

REM === Убираем first-run email prompt ===
REM Streamlit при первом запуске ждёт ввода email и блокирует старт сервера.
REM Создаём конфиг-файл, который отключает это поведение.
set STREAMLIT_DIR=%USERPROFILE%\.streamlit
if not exist "%STREAMLIT_DIR%" mkdir "%STREAMLIT_DIR%"

REM Если credentials.toml не существует — создаём
if not exist "%STREAMLIT_DIR%\credentials.toml" (
    echo [general]> "%STREAMLIT_DIR%\credentials.toml"
    echo email = "">> "%STREAMLIT_DIR%\credentials.toml"
    echo Streamlit credentials configured ^(no email prompt^).
)

REM Если config.toml не существует — создаём с базовыми настройками
if not exist "%STREAMLIT_DIR%\config.toml" (
    echo [browser]> "%STREAMLIT_DIR%\config.toml"
    echo gatherUsageStats = false>> "%STREAMLIT_DIR%\config.toml"
    echo.>> "%STREAMLIT_DIR%\config.toml"
    echo [server]>> "%STREAMLIT_DIR%\config.toml"
    echo headless = true>> "%STREAMLIT_DIR%\config.toml"
    echo Streamlit config created.
)

echo.
echo Запуск Streamlit UI на http://localhost:8501
echo Браузер откроется автоматически через несколько секунд...
echo.

REM Запускаем открытие браузера с задержкой 5 секунд в фоне
start /b cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"

REM Запускаем streamlit (headless — не пытается сам открыть браузер,
REM мы это делаем выше с задержкой)
python -m streamlit run interfaces/web.py ^
    --server.address 127.0.0.1 ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false

echo.
echo Streamlit завершился.
pause
