def scoring(score, parameter):
    if score > 0.98:
        parameter += 5
    
    elif score > 0.95:
        parameter += 4
    
    elif score > 0.92:
        parameter += 3
    
    elif score > 0.90:
        parameter += 2
    
    elif score > 0.85:
        parameter += 1
    
    return parameter