@echo off
echo ========================================
echo Сборка J.A.R.V.I.S. в exe (UAC Admin)
echo ========================================
echo.

:: Удаляем старую папку dist
if exist dist (
    echo Удаление старой сборки...
    rmdir /s /q dist
)

:: Удаляем папку build
if exist build (
    rmdir /s /q build
)

:: Удаляем старый spec
if exist JARVIS.spec (
    del JARVIS.spec
)

:: Собираем через PyInstaller
echo Запуск PyInstaller...
python -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --name=JARVIS ^
    --onedir ^
    --windowed ^
    --uac-admin ^
    --add-data "config.json;." ^
    --exclude-module=PyQt5 ^
    --exclude-module=PyQt6 ^
    --exclude-module=PySide2 ^
    --exclude-module=matplotlib ^
    --exclude-module=scipy ^
    --exclude-module=torch ^
    --exclude-module=torchaudio ^
    --exclude-module=pandas ^
    --exclude-module=IPython ^
    --exclude-module=notebook ^
    --exclude-module=pytest ^
    main.py

echo.
if exist dist\JARVIS\JARVIS.exe (
    echo ========================================
    echo Сборка завершена успешно!
    echo ========================================
    echo.
    echo Исполняемый файл: dist\JARVIS\JARVIS.exe
    echo Папка сборки:     dist\JARVIS\_internal\
    echo.
    echo ВАЖНО: файл требует прав администратора!
    echo 1. Нажмите правой кнопкой на JARVIS.exe
    echo 2. Выберите "Запуск от имени администратора"
    echo.
    echo Для автозапуска (требует админ-прав):
    echo    1. Win+R -> shell:startup -> Enter
    echo    2. Создайте ярлык на JARVIS.exe
    echo    3. Свойства ярлыка -> Дополнительно -> "Запуск от имени администратора"
    echo.
    echo Настройки хранятся в: dist\JARVIS\config.json
    echo Чаты хранятся в:       dist\JARVIS\chats\
    echo Логи хранятся в:       dist\JARVIS\logs\
) else (
    echo ========================================
    echo Ошибка сборки!
    echo ========================================
)
echo.
pause