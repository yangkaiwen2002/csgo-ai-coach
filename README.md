# CS:GO AI Coach 🎯

An AI-powered CS:GO replay analyzer that uses Deep Learning to simulate game situations and recommend optimal decisions for players.

## 💡 Core Idea

Given **limited information** (what a real player can actually see/hear), the model learns from **pro player demos** and tells you:
> "In this situation, the best move was X — here's why."

## 🧠 How It Works

```
.dem replay file
      ↓
  Demo Parser (awpy)
      ↓
  Feature Extractor (limited-info perspective)
      ↓
  Decision Model (Transformer/LSTM)
      ↓
  "You should have pushed A / held B / thrown smoke..."
```

## 🚀 Features (Roadmap)

- [x] Demo parsing pipeline
- [x] Feature extraction (player-perspective only)
- [x] Decision classification model
- [ ] Web UI for replay upload & visualization
- [ ] Real-time overlay (future)

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/csgo-ai-coach.git
cd csgo-ai-coach
pip install -r requirements.txt
```

## 🎮 Quick Start

### 1. Parse a demo file
```bash
python scripts/parse_demo.py --demo path/to/your.dem --output data/parsed/
```

### 2. Extract features
```bash
python scripts/extract_features.py --input data/parsed/ --output data/features/
```

### 3. Train the model
```bash
python scripts/train.py --config configs/model_config.yaml
```

### 4. Analyze your replay
```bash
python scripts/analyze.py --demo path/to/your.dem --model checkpoints/best_model.pt
```

## 📁 Project Structure

```
csgo-ai-coach/
├── src/
│   ├── data/           # Demo parsing & dataset
│   ├── features/       # Feature engineering
│   ├── models/         # ML models
│   └── analysis/       # Replay analysis & reporting
├── scripts/            # CLI entry points
├── configs/            # Model & training configs
├── notebooks/          # Exploration notebooks
└── tests/              # Unit tests
```

## 🔧 Data Sources

- Download pro demos: [HLTV.org](https://www.hltv.org/matches) (free)
- Uses `awpy` for `.dem` file parsing

## 📊 Model

- **Input**: Game state at each round tick (limited-info perspective)
- **Output**: Action recommendation (Push A / Push B / Hold / Smoke / Fall back)
- **Architecture**: Transformer encoder over time-series game states

## 🤝 Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
