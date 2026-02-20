#!/usr/bin/env bash
# Start both Flask backend and React frontend

echo "🚀 Starting RAG Q&A System"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Start backend
echo -e "${BLUE}Starting Flask backend...${NC}"
cd backend
python app.py &
BACKEND_PID=$!
echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"
echo ""

# Wait a moment for backend to start
sleep 3

# Start frontend
echo -e "${BLUE}Starting React frontend...${NC}"
cd ../frontend
npm start &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
echo ""

echo "=========================================="
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
