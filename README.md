# EcoPackAI – Backend (Flask + ML + PostgreSQL)

EcoPackAI Backend is a Flask-based REST API that predicts **CO₂ emission**, **cost**, and provides an **AI-based eco-friendly material recommendation** for packaging products.  
The backend integrates **Machine Learning models** with a **PostgreSQL database** and exposes APIs for frontend/UI consumption.

---

## 🚀 Features

- REST API built using **Flask**
- AI-based **material recommendation**
- **CO₂ emission prediction**
- **Cost prediction**
- **Environmental score computation**
- PostgreSQL database integration
- JSON-based secure API responses
- GitHub collaboration-ready backend

---

## 🛠️ Tech Stack

- **Backend Framework:** Flask (Python)
- **Machine Learning:** Scikit-learn (RandomForest models)
- **Database:** PostgreSQL
- **ORM/DB Connector:** psycopg2
- **API Testing:** Thunder Client / Postman
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

EcoPackAI_Backend/
│
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
├── .gitignore
├── model/
│ ├── rf_co2.pkl # CO₂ prediction model
│ └── rf_cost.pkl # Cost prediction model
├── data/ # (optional) datasets
└── venv/ # Virtual environment (ignored in Git)

---

## ⚙️ Setup Instructions
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/debasri-pal/EcoPackAI_Backend.git
cd EcoPackAI_Backend
2️⃣ Create Virtual Environment

python -m venv venv
Activate it:

Windows = venv\Scripts\activate

Mac/Linux = source venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt
4️⃣ PostgreSQL Setup
Create database:

CREATE DATABASE ecopackai;
Tables used:

products
predictions

5️⃣ Run the Flask App

python app.py
Server runs on:  http://127.0.0.1:5000
📌 API Endpoints
🔹 Health Check
GET /

{
  "message": "EcoPackAI Backend is running 🚀"
}
🔹 AI Material Recommendation
POST /recommend-material

Request Body (JSON):

{
  "product_name": "Food Box",
  "material_type": "Paper",
  "weight": 1.2,
  "volume": 3.5,
  "recyclable": true
}
Response (JSON):


{
  "product": "Food Box",
  "co2_prediction": 83.72,
  "cost_prediction": 2.8,
  "environmental_score": 30.14,
  "recommended_material": "Traditional Plastic"
}
🗄️ Database Verification (Optional)
Run in pgAdmin Query Tool:

SELECT * FROM products;
SELECT * FROM predictions;
🤝 Collaboration Workflow
Backend code is hosted on GitHub


Frontend team can pull APIs and integrate UI

All development follows Git-based collaboration

📈 Current Status
✅ Backend API completed
✅ ML model integration done
✅ PostgreSQL integration done
✅ API tested successfully
⏳ Frontend & UI integration in progress

👩‍💻 Author
Debasri Pal
B.Tech CSE
Backend Developer – EcoPackAI Project

📜 License
This project is for academic and educational purposes.
---
