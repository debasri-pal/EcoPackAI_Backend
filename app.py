from flask import Flask, request, jsonify
import joblib
import numpy as np
import psycopg2
import os

app = Flask(__name__)

# -----------------------
# Load ML models
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_co2 = joblib.load(os.path.join(BASE_DIR, "model", "rf_co2.pkl"))
rf_cost = joblib.load(os.path.join(BASE_DIR, "model", "rf_cost.pkl"))

print("✅ Models loaded successfully")

# -----------------------
# PostgreSQL connection
# -----------------------
def get_db_connection():
    return psycopg2.connect(
        dbname="ecopackai",
        user="postgres",
        password="tiya@780Dp",
        host="localhost",
        port="5432"
    )

# -----------------------
# Health check
# -----------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "EcoPackAI Backend is running 🚀"})

# -----------------------
# AI Material Recommendation API
# -----------------------
@app.route("/recommend-material", methods=["POST"])
def recommend_material():
    data = request.json

    # ---------- STEP 1: Encode inputs ----------
    material_map = {
        "Paper": 1,
        "Plastic": 2,
        "Glass": 3,
        "Metal": 4
    }

    product_name = data["product_name"]
    material_type = data["material_type"]
    weight = float(data["weight"])
    volume = float(data["volume"])
    recyclable = data["recyclable"]

    material_encoded = material_map.get(material_type, 0)
    recyclable_encoded = 1 if recyclable else 0

    density = 2.5  # used during training

    features = np.array([
        material_encoded,
        recyclable_encoded,
        weight,
        volume,
        density
    ]).reshape(1, -1)

    # ---------- STEP 2: Predictions ----------
    co2 = rf_co2.predict(features)[0]
    cost = rf_cost.predict(features)[0]

    # ✅ FIX: convert NumPy → Python floats (VERY IMPORTANT)
    co2 = float(co2)
    cost = float(cost)

    environmental_score = max(0, 100 - (co2 * 0.5 + cost * 10))
    environmental_score = float(environmental_score)

    # ---------- STEP 3: Recommendation ----------
    if environmental_score >= 70:
        recommendation = "Recycled Paper"
    elif environmental_score >= 50:
        recommendation = "Biodegradable Plastic"
    else:
        recommendation = "Traditional Plastic"

    # ---------- STEP 4: Save to DB ----------
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO products (product_name, material_type, weight, volume, recyclable)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """, (product_name, material_type, weight, volume, recyclable))

    product_id = cur.fetchone()[0]

    cur.execute("""
        INSERT INTO predictions (product_id, co2_prediction, cost_prediction, environmental_score)
        VALUES (%s, %s, %s, %s)
    """, (product_id, co2, cost, environmental_score))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "product": product_name,
        "co2_prediction": round(co2, 2),
        "cost_prediction": round(cost, 2),
        "environmental_score": round(environmental_score, 2),
        "recommended_material": recommendation
    })


# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
