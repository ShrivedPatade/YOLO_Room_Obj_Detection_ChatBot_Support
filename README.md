
# Embodied Smart Home Agent (Eyes-Brain Architecture)

An asynchronous, Embodied Smart Home AI that continuously monitors physical spaces using local computer vision and responds to natural language queries using a persistent memory bank.

This project decouples the "Eyes" (Vision Daemon) from the "Brain" (Chat Agent). The Vision system acts as a background multiplexer, constantly sweeping video feeds and logging spatial coordinates. The Brain acts as an on-demand LangGraph state machine, reading the memory bank to provide highly accurate, hallucination-free answers about object locations.

## 🏗️ Architecture

1. **Fast Eyes (Vision Daemon):** Uses `YOLO-World` (Zero-shot open-vocabulary detection) via OpenCV to scan continuous video feeds or simulated MP4s, saving object locations and timestamps to a persistent JSON memory bank.
2. **Memory Bank:** A `memory_bank.json` file acting as the shared state between the asynchronous processes.
3. **The Brain (Chat Agent):** A `LangGraph` workflow that parses user intent (via local Ollama models like Qwen), aggregates the Python dictionary memory by room, and utilizes the `Gemini 1.5 Flash` API to naturally relay physical facts to the user.

## 📂 Project Structure

```text
YOLO_Room_Obj_Detection_ChatBot_Support/
│
├── main.py                     # Entry point for the Chat Agent
├── vision_daemon.py            # Background daemon for continuous room scanning
├── vision_video_daemon.py      # Background daemon for MP4 simulated testing
├── config.py                   # Central vocabulary and configurations
├── requirements.txt            # Python dependencies
│
├── vision/
│   └── scanner.py              # YOLO-World CV2 frame processing logic
│   └── spatial.py              # Spatial Calculations
│
└── cognition/
    └── agent.py                # LangGraph nodes, Memory Fetching, and Gemini generation
    └── state.py                # Define States

```

## ⚙️ Setup & Installation

**1. Clone the repository**

```bash
git clone [https://github.com/ShrivedPatade/YOLO_Room_Obj_Detection_ChatBot_Support.git](https://github.com/ShrivedPatade/YOLO_Room_Obj_Detection_ChatBot_Support.git)
cd YOLO_Room_Obj_Detection_ChatBot_Support

```

**2. Initialize the Virtual Environment**

```bash
python -m venv env
env\Scripts\activate   # Windows
# source env/bin/activate # Mac/Linux

```

**3. Install Dependencies**

```bash
pip install -r requirements.txt

```

**4. Download Vision Models**
Ensure you have the YOLO-World model downloaded in your root directory.

* Download: `yolov8l-world.pt` (Large model recommended for accuracy).

**5. Configure Environment Variables**
Create a `.env` file in the root directory to store your cloud API keys securely:

```env
GEMINI_API_KEY="your_google_gemini_api_key"

```

*Note: Ensure you have [Ollama](https://ollama.com/) installed and running locally with the `qwen2.5:1.5b` model pulled (`ollama pull qwen2.5:1.5b`) for the intent parsing node.*

## 🚀 Running Inference (The Dual-Terminal Approach)

Because this is a true embodied agent, it requires two concurrent processes to run: the daemon to watch the space, and the agent to talk to you.

**Terminal 1: Start the Vision Daemon**
Activate your environment and start the continuous camera scanner (or the video simulator if testing with videos).

```bash
python vision_video_daemon.py

```

*Leave this running. You will see it logging updates to the `memory_bank.json` every few seconds.*

**Terminal 2: Start the Chat Agent**
Open a new terminal, activate the environment, and run the LangGraph chatbot.

```bash
python main.py

```

### Example Interaction

**User:** "Where is the TV, the bowl, and my keys?"

**Agent Execution:**

1. *Intent Parse (Local Qwen):* `["tv", "bowl", "keys"]`
2. *Memory Fetch (Python):* Scans `memory_bank.json`. Finds TV in Living Room, Bowl in Kitchen. Fails to find keys.
3. *Generation (Gemini 2.5 Flash):* Reads the objective memory string and generates output.

**Agent Output:** > "The TV is located in the Living Room (last seen at 2026-05-13 14:22:38), and the bowl is in the Kitchen at the same time. The keys are missing from memory."

## 🛠️ Customization

To add or remove trackable objects across the entire system, simply edit the `TRACKABLE_OBJECTS` list inside `config.py`. The YOLO-World model will dynamically generate the new text-embeddings on the fly, and the Intent Parser will strictly validate against your updated vocabulary.

```

```
