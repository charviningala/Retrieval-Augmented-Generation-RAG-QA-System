@echo off
REM Start Flask backend which serves both API and frontend

echo.
echo 🚀 Starting RAG Q&A System (Integrated)
echo.

cd backend
echo Starting backend with integrated frontend...
python app.py

echo.
echo =========================================
echo Application: http://localhost:5000
echo =========================================
echo.
echo Press Ctrl+C to stop the server
echo.
pause
