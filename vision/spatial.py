def calculate_center(bbox):
    """Calculates the (x, y) center of a bounding box [xmin, ymin, xmax, ymax]."""
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return x_center, y_center

def is_inside(boxA, boxB):
    """Checks if Box A is mostly inside Box B."""
    # A simple check: are A's min/max bounds mostly within B's?
    if boxA[0] >= boxB[0] and boxA[2] <= boxB[2] and \
       boxA[1] >= boxB[1] and boxA[3] <= boxB[3]:
        return True
    return False

def get_spatial_relationship(objA, objB):
    """
    Compares two objects (dictionaries with 'coordinates') 
    and returns a natural language relationship.
    """
    boxA = objA['coordinates']
    boxB = objB['coordinates']
    
    if is_inside(boxA, boxB):
        return f"inside the {objB['object']}"
    
    # Calculate centers
    cx_A, cy_A = calculate_center(boxA)
    cx_B, cy_B = calculate_center(boxB)
    
    # Determine horizontal and vertical relative positions
    horizontal = "left of" if cx_A < cx_B else "right of"
    vertical = "above" if cy_A < cy_B else "below"
    
    # Decide which axis has the more dominant difference
    x_diff = abs(cx_A - cx_B)
    y_diff = abs(cy_A - cy_B)
    
    if x_diff > y_diff:
        return horizontal
    else:
        return vertical