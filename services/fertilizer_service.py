import numpy as np

def get_fertilizer_recommendation(model, temperature, humidity, moisture, soil_type_enc, crop_type_enc, n, p, k, land_area):
    """Returns fertilizer name, computed quantity per acre, total quantity, and agronomic explanation."""
    input_data = np.array([[temperature, humidity, moisture, soil_type_enc, crop_type_enc, n, k, p]])
    prediction = model.predict(input_data)[0]
    
    # Map raw model output (which might be numeric like 28-28 or generic text) to real-world names
    raw_pred = str(prediction).strip().upper()
    
    fertilizer_name = raw_pred
    explanation = ""
    
    # Heuristics to map to standard fertilizers if the model outputs ratios
    if "28" in raw_pred or "28-28" in raw_pred:
        fertilizer_name = "NPK 28-28-0"
    elif "14" in raw_pred:
        fertilizer_name = "NPK 14-35-14"
    elif "10" in raw_pred:
        fertilizer_name = "NPK 10-26-26"
    elif "UREA" in raw_pred:
        fertilizer_name = "Urea (46-0-0)"
    elif "DAP" in raw_pred:
        fertilizer_name = "DAP (18-46-0)"

    # Base Application Logic
    base_qty_per_acre = 50 
    
    if n < 30 and k > 50 and p > 50:
        fertilizer_name = "Urea (46-0-0)"
        explanation = "Nitrogen is critically low in the soil while Potassium and Phosphorus are sufficient. Urea is highly recommended to rapidly increase leaf and stem growth."
        base_qty_per_acre = 75
        
    elif p < 30 and n > 40:
        fertilizer_name = "DAP (18-46-0)"
        explanation = "Phosphorus is low, which is vital for root development. DAP provides a concentrated dose of Phosphorus."
        base_qty_per_acre = 100
        
    elif k < 30:
        fertilizer_name = "MOP (Muriate of Potash 0-0-60)"
        explanation = "Potassium is deficient. MOP is recommended to improve disease resistance and water retention."
        base_qty_per_acre = 40
        
    else:
        # Default balanced explanation based on whatever the model spat out
        if "UREA" in fertilizer_name.upper():
            explanation = "Urea is recommended to supply the necessary Nitrogen payload for your crop type."
            base_qty_per_acre = 60
        elif "DAP" in fertilizer_name.upper():
            explanation = "DAP is recommended to balance out early-stage root development."
            base_qty_per_acre = 80
        elif "NPK" in fertilizer_name.upper():
            explanation = f"A balanced {fertilizer_name} blend is optimal to maintain even nutrient distribution across your field."
            base_qty_per_acre = 50
    
    total_qty = base_qty_per_acre * land_area
    
    return {
        "fertilizer": fertilizer_name,
        "qty_per_acre": base_qty_per_acre,
        "total_qty": total_qty,
        "explanation": explanation
    }
