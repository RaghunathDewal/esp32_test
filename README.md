# gemini-esp32-bridge

Testing-level WebSocket bridge between an ESP32 and the Gemini Live API.

```
ESP32 (mic/speaker) <--WebSocket--> this server (Render) <--WebSocket--> Gemini Live API
```

The ESP32 streams raw 16-bit PCM mono audio (16kHz) to the server as binary
WebSocket frames. The server forwards it to Gemini Live, and streams
Gemini's spoken response (24kHz PCM16) back to the ESP32 as binary frames.

## Folder structure

```
gemini-esp32-bridge/
├── server.py         # FastAPI WebSocket server (the bridge)
├── tools.py           # Test tools (get_current_time, add_numbers) shared with Gemini
├── test_client.py     # Simulates an ESP32 using your PC mic/speakers, for testing
├── requirements.txt   # Python dependencies
├── render.yaml         # Render deployment config
├── .env.example        # Environment variable template
└── README.md
```

## 1. Local setup

```bash
cd gemini-esp32-bridge
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then fill in GEMINI_API_KEY
```

Run the server locally:

```bash
export GEMINI_API_KEY=your_key_here   # Windows PowerShell: $env:GEMINI_API_KEY="your_key_here"
python server.py
```

The server listens on `ws://localhost:8000/ws` (health check at `GET /`).

## 2. Test before touching the ESP32

Use `test_client.py` to simulate an ESP32 using your PC's mic/speakers —
this confirms the server-to-Gemini path works before you deal with firmware:

```bash
pip install pyaudio websockets
python test_client.py ws://localhost:8000/ws
```

Speak into your mic; you should hear Gemini's voice reply through your
speakers, and see tool-call logs in the server terminal if you ask it
something like "what's 12 plus 30?".

## 3. Deploy to Render

1. Push this folder to a GitHub repo.
2. In Render: **New +** → **Blueprint** → point at the repo (it will pick
   up `render.yaml` automatically), or create a **Web Service** manually with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
3. Set the `GEMINI_API_KEY` environment variable in the Render dashboard
   (Settings → Environment).
4. Deploy. Your WebSocket endpoint will be:
   `wss://<your-app-name>.onrender.com/ws`

Note: Render's free tier spins down on inactivity, so the first connection
after idling will be slow (cold start) — fine for testing, not for production.

## 4. ESP32 integration

On the ESP32 side, use a WebSocket client library (e.g. `arduinoWebSockets`
or ESP-IDF's `esp_websocket_client`) to:

1. Connect to `wss://<your-app-name>.onrender.com/ws`.
2. Continuously send binary frames of raw 16-bit PCM mono audio at 16kHz
   captured from an I2S mic (e.g. INMP441).
3. On receiving binary frames from the server, play them out via I2S to a
   speaker/amp (24kHz PCM16).

Keep ESP32-side chunk sizes small (e.g. 512–1024 bytes) to keep latency low.
This server does no framing/encoding beyond raw PCM — match sample rate and
bit depth exactly on the firmware side or audio will sound distorted/pitched
wrong.

## Notes / limitations (testing-level only)

- No authentication on the WebSocket endpoint — anyone with the URL can
  connect and use your Gemini API quota. Add a token check before real use.
- No reconnection/backoff logic on either side.
- One Gemini Live session per WebSocket connection; each ESP32 device
  should open its own connection.
