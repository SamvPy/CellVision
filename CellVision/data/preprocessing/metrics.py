import cv2
import numpy as np

def contour_metrics(contour):
   
    if contour is None:
        return {
            "area": np.nan,
            "perimeter": np.nan,
            "circularity": np.nan,
            "equivalent_diameter": np.nan,
            "aspect_ratio": np.nan,
            "elongation": np.nan,
            # "eccentricity": np.nan,
            "solidity": np.nan,
            "convexity": np.nan,
        }

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    # ellipse
    if len(contour) > 5:
        (xc, yc), (MA, ma), angle = cv2.fitEllipse(contour)
        elongation =  MA / ma
    else:
        MA, ma = np.nan, np.nan
        elongation = np.nan

    metrics = {
        "area": area,
        "perimeter": perimeter,
        "circularity": 4*np.pi*area / (perimeter*perimeter),
        "equivalent_diameter": np.sqrt(4*area/np.pi),
        "aspect_ratio": None,
        "elongation": elongation,
        # "eccentricity": np.sqrt(1 - (ma/MA)**2),
        "solidity": area / hull_area,
        "convexity": hull_perimeter / perimeter,
    }

    # bounding box–based
    x, y, w, h = cv2.boundingRect(contour)
    metrics["aspect_ratio"] = w / h
    metrics["extent"] = area / (w * h)

    return metrics