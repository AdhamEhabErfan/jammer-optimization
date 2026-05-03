# Jammer Optimization using Neural Networks

## 📌 Overview

This project focuses on optimizing a wireless jammer against a frequency-hopping communication system using machine learning techniques.

The system learns to predict frequency hopping patterns and allocate jamming power efficiently over time.

---

## 🧠 Methods Used

* LSTM (sequence learning for frequency prediction)
* Deep Q-Network (DQN) for power allocation
* Hybrid neural network combining prediction + optimization

---

## ⚙️ Features

* Simulated frequency-hopping transmitter (random, Markov, chaotic)
* Reinforcement learning environment for jammer training
* Adaptive power allocation strategy
* Performance evaluation and visualization tools

---

## 🚀 How to Run

### Install dependencies

pip install torch numpy matplotlib gym scipy

### Train

python main.py --mode train --algorithm markov

### Evaluate

python main.py --mode evaluate

### Visualize

python main.py --mode visualize

---

## 📊 Results

The model significantly improves jamming success rate compared to random and uniform strategies, especially against structured hopping algorithms.

---

## 📁 Project Structure

* config.py → parameters
* train.py → training pipeline
* models/ → neural networks
* jammer_environment.py → RL environment
* evaluate.py → testing
* visualize.py → plots

---

## ⚠️ Notes

* Trained models (.pth) are not included in the repository
* You can train them locally or on cloud (e.g., Google Colab)

---

## 🎯 Future Work

* Transformer-based prediction
* Multi-agent jamming
* Real SDR integration
* Game-theoretic adaptive transmitter

---
