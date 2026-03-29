def validate_and_clip_crop_inputs(n, p, k, ph, temperature, humidity, rainfall):
    """
    Validates inputs. If values are out of physical bounds, clips them to defaults and returns warnings
    so the model can still predict instead of failing outright.
    """
    warnings = []
    
    # Clip Temperature
    if temperature < 5:
        warnings.append(f"Temperature ({temperature}°C) is unusually low. Adjusted to minimum valid threshold (5°C).")
        temperature = 5
    elif temperature > 50:
        warnings.append(f"Temperature ({temperature}°C) is unusually high. Adjusted to maximum valid threshold (50°C).")
        temperature = 50
        
    # Clip Humidity
    if humidity < 10:
        warnings.append(f"Humidity ({humidity}%) is unusually low for crop growth. Adjusted to minimum valid threshold (10%).")
        humidity = 10
    elif humidity > 100:
        warnings.append(f"Humidity ({humidity}%) is invalid. Adjusted to 100%.")
        humidity = 100
        
    # Clip Rainfall
    if rainfall < 0:
        warnings.append(f"Rainfall ({rainfall}mm) cannot be negative. Adjusted to 0mm.")
        rainfall = 0
        
    # Clip pH
    if ph < 3:
        warnings.append(f"Soil pH ({ph}) is too acidic. Adjusted to minimum valid threshold (3).")
        ph = 3
    elif ph > 10:
        warnings.append(f"Soil pH ({ph}) is too alkaline. Adjusted to maximum valid threshold (10).")
        ph = 10
        
    # Clip Nutrients
    if n < 0: n = 0
    if p < 0: p = 0
    if k < 0: k = 0
        
    return temperature, humidity, rainfall, ph, n, p, k, warnings

def validate_and_clip_fertilizer_inputs(n, p, k, temperature, humidity, moisture, land_area):
    """Clips inputs for fertilizer logic gracefully."""
    warnings = []
    
    if temperature < 5:
        temperature = 5
    elif temperature > 50:
        temperature = 50
        
    if humidity < 10: humidity = 10
    elif humidity > 100: humidity = 100
        
    if moisture < 0: moisture = 0
    elif moisture > 100: moisture = 100
        
    if n < 0: n = 0
    if p < 0: p = 0
    if k < 0: k = 0
    
    if land_area <= 0:
        warnings.append("Land area must be greater than 0. Defaulting to 1.")
        land_area = 1.0
        
    return temperature, humidity, moisture, n, p, k, land_area, warnings
