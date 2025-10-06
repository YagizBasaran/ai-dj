#!/bin/bash

# Write Firebase config from env var to file if it exists
if [ ! -z "$FIREBASE_CONFIG" ]; then
    echo "$FIREBASE_CONFIG" > /app/firebase-key.json
    echo "✅ Firebase credentials configured"
fi

# Start gunicorn on the port Render provides
exec gunicorn -w 2 -b 0.0.0.0:$PORT app:app