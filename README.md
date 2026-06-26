# Explainable-Federated-Digital-Twin-Based-Predictive-Maintenance-for-Smart-Manufacturing

output :-
<img width="1918" height="1087" alt="image" src="https://github.com/user-attachments/assets/86fc83a8-7fed-4f2d-aa56-a95047e292f4" />
<img width="1907" height="1067" alt="image" src="https://github.com/user-attachments/assets/4cbf2f5b-db6b-4ba5-a1e0-e5116436f167" />
<img width="1918" height="1022" alt="image" src="https://github.com/user-attachments/assets/7d6e386d-c60c-42fc-a5ed-4f89e700c61a" />
<img width="1915" height="1080" alt="image" src="https://github.com/user-attachments/assets/9e2874e0-ee33-46cb-9da8-01ed6aead2e3" />


Here is a production-ready, professional README.md template written for this exact repository. You can copy and paste this directly into the project.

Explainable Federated Digital Twin-Based Predictive Maintenance (EF-DT-PdM)
An open-source, privacy-preserving AI pipeline for proactive fault detection, root-cause explainability, and automated repair guidance in Industry 4.0.

📖 Overview
In modern smart manufacturing, predictive maintenance (PdM) saves millions in downtime, but faces three major bottlenecks:

The Privacy Bottleneck: Factories refuse to upload highly sensitive, proprietary machine telemetry to centralized cloud servers.

The "Black Box" Bottleneck: Operators do not trust an AI that simply shouts "Machine failing in 4 hours" without explaining why.

The Execution Bottleneck: Knowing a machine is breaking is useless if the junior technician on duty doesn’t know how to fix it.

This project solves all three by uniting Federated Learning, Digital Twins, Explainable AI (XAI), and Retrieval-Augmented Generation (RAG) into a single end-to-end workflow.

✨ Key Features
🏭 Live Digital Twin Telemetry: Ingests real-time multi-sensor streams (vibration, thermal, acoustic) and maps them to a virtual instance of the machinery.

🔒 Zero-Data-Leakage Federated Learning: Edge nodes train local failure-forecasting models. Only encrypted mathematical weight updates are sent to the central aggregator; raw sensor data never leaves the factory floor.

🔬 SHAP/LIME Explainability: Deconstructs anomaly alerts into human-readable feature attributions (e.g., "Failure probability 88%: driven by a 4.2x spike in Axis-Z vibration over the last 12 rolling window cycles").

📚 Autonomous RAG Technician Assistant: Automatically queries a vectorized database of complex equipment PDF manuals to generate a step-by-step repair guide tailored to the specific XAI diagnosis.

🏛️ System Architecture
Plaintext
  [ Physical Machine ] ──( Live Sensor Telemetry )──► [ Digital Twin UI ]
                                                              │
  [ Local Edge Model ] ◄──( Trains Locally )──────────────────┘
           │
     (Model Weights)
           ▼
  [ Central Aggregator ] (Federated Averaging)
           │
     (Anomaly Flagged)
           ▼
    [ XAI Engine ] ──────► "Bearing #4 Overheating" 
                                  │
                                  ▼
    [ RAG Pipeline ] ────► (Searches PDF Manuals) ───► Outputs Step-by-Step Fix
📂 Repository Structure
Plaintext
├── data/                  # Sample / simulated digital twin sensor logs
├── rag/
│   ├── manuals/           # Raw equipment PDF technical documentation
│   └── vectorstore/       # Persistent ChromaDB/FAISS vector embeddings
├── src/
│   ├── models/            # Federated client/server logic & base classifiers
│   ├── preprocessing/     # Rolling averages, Fourier transforms, cleaning scripts
│   └── xai/               # SHAP/LIME wrapper functions 
├── dashboard/
│   └── app.py             # Main Streamlit web application
├── requirements.txt       # Python environment dependencies
└── README.md
🛠️ Tech Stack
Core Machine Learning: scikit-learn, NumPy, Pandas

Federated Infrastructure: Custom Python orchestration / Flower (flwr)

Explainable AI: shap, lime

Contextual AI (RAG): LangChain, ChromaDB, OpenAI API / Ollama

Frontend Dashboard: Streamlit, Plotly

🚀 Getting Started
1. Prerequisites
Ensure you have Python 3.10+ installed.

2. Installation
Clone the repository and install the required packages:

Bash
git clone https://github.com/Shubham-Ut/Explainable-Federated-Digital-Twin-Based-Predictive-Maintenance-for-Smart-Manufacturing.git
cd Explainable-Federated-Digital-Twin-Based-Predictive-Maintenance-for-Smart-Manufacturing

pip install -r requirements.txt
3. Setup the Knowledge Base (RAG)
Place your raw machinery user manuals (PDFs) inside rag/manuals/, then build the vector database:

Bash
python src/rag/embed_manuals.py
4. Launch the Digital Twin Dashboard
Spin up the live interactive UI:

Bash
streamlit run dashboard/app.py
🖥️ User Workflow inside the Dashboard
Select Node: Choose Factory A - CNC Mill #4 from the sidebar.

Monitor: Watch the live streaming telemetry graphs.

Simulate Degradation: Inject an artificial failure signature (e.g., raise friction coefficients).

Inspect the Alert: View the generated SHAP waterfall plot showing which sensor tipped the algorithm over the safe threshold.

Get the Fix: Look at the bottom pane—the RAG module will display the exact page, diagram, and 5-step wrench procedure to replace that specific bearing.

📜 License
Distributed under the MIT License. See LICENSE for more information.
