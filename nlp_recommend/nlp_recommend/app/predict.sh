#!/usr/bin/env bash

PORT=5000
echo "Port: $PORT"

# POST method predict
curl -d '{  
   "thoughts":"I want to practice more sport"
}'\
     -H "Content-Type: application/json" \
     -X POST http://localhost:$PORT/