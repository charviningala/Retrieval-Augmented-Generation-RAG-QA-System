@echo off
REM Start both Flask backend and React frontend on Windows

echo.
echo 🚀 Starting RAG Q&A System
echo.

REM Start backend
echo Starting Flask backend...
cd backend
start cmd /k "python app.py"
echo Backend started in new window
timeout /t 3 /nobreak

REM Start frontend
cd ..
cd frontend
echo Starting React frontend...
start cmd /k "npm start"
echo Frontend started in new window

echo.
echo =========================================
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:3000
echo =========================================
echo.
echo You can close this window. Use Ctrl+C in each window to stop services.
echo.
pause
