import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load saved components
model = joblib.load("purchase_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_columns.pkl")

def predict_purchase(
    age,
    gender,
    annual_income,
    number_of_purchases,
    product_category,
    time_spent,
    loyalty_program,
    discounts_availed
):
    """
    Predict whether a customer will make a purchase.
    
    Args:
        age: Customer's age
        gender: Gender (0 = Male, 1 = Female)
        annual_income: Annual income of customer
        number_of_purchases: Total purchases made
        product_category: Category of product
        time_spent: Time spent on website (minutes)
        loyalty_program: Loyalty program membership (0/1)
        discounts_availed: Number of discounts used (0-5)
    
    Returns:
        Prediction result with confidence visualization
    """
    try:
        # Validate inputs
        if age <= 0 or age > 120:
            return "⚠️ Error: Please enter a valid age (1-120)"
        if annual_income < 0:
            return "⚠️ Error: Annual income cannot be negative"
        if time_spent < 0:
            return "⚠️ Error: Time spent cannot be negative"
        if number_of_purchases < 0:
            return "⚠️ Error: Number of purchases cannot be negative"
        
        # Create input dataframe
        input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "annual_income": annual_income,
        "number_of_purchases": number_of_purchases,
        "product_category": product_category,
        "time_spent_on_website": time_spent,
        "loyalty_program": loyalty_program,
        "discounts_availed": discounts_availed
        }])
    

        # Ensure correct column order
        input_data = input_data[features]
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Get prediction and probability
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Format result
        if prediction == 1:
            confidence = probability[1] * 100
            result = f"✅ Purchase Likely\nConfidence: {confidence:.2f}%"
        else:
            confidence = probability[0] * 100
            result = f"❌ Purchase Unlikely\nConfidence: {confidence:.2f}%"
        
        return result
    
    except Exception as e:
        return f"⚠️ Error in prediction: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Number(label="👤 Age", value=25, minimum=1, maximum=120),
        gr.Radio([0, 1], label="👥 Gender", value=0, info="0 = Male, 1 = Female"),
        gr.Number(label="💰 Annual Income ($)", value=50000, minimum=0),
        gr.Number(label="🛍️ Number of Purchases", value=5, minimum=0),
        gr.Radio(
            [0, 1, 2, 3, 4],
            label="📦 Product Category",
            value=0,
            info="0=Electronics, 1=Clothing, 2=Home, 3=Beauty, 4=Sports"
        ),
        gr.Number(label="⏱️ Time Spent on Website (minutes)", value=30, minimum=0),
        gr.Radio([0, 1], label="⭐ Loyalty Program", value=0, info="0 = No, 1 = Yes"),
        gr.Number(label="🎟️ Discounts Availed", value=2, minimum=0, maximum=5)
    ],
    outputs=gr.Textbox(label="📊 Prediction Result",lines= 4),
    title="🎯 Customer Purchase Prediction System",
    description="""
    <div style="text-align:center; font-size:16px;">
        Predict whether a customer is likely to make a purchase using a trained ML model.
    </div>
    """,
    submit_btn="Predict",
    theme=gr.themes.Soft(),
    examples=[
        [25, 0, 50000, 5, 0, 30, 0, 2],
        [35, 1, 75000, 10, 1, 45, 1, 3],
        [55, 0, 120000, 20, 2, 60, 1, 5],
    ]
)

if __name__ == "__main__":
    interface.launch()