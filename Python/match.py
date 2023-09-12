def matched(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2, threshold=0.65):
    intersection_min_x = max(min_x1, min_x2)
    intersection_min_y = max(min_y1, min_y2)
    intersection_max_x = min(max_x1, max_x2)
    intersection_max_y = min(max_y1, max_y2)
    print('intersection')
    print(intersection_min_x)
    print(intersection_min_y)
    print(intersection_max_x)
    print(intersection_max_y)
    if intersection_max_x <= intersection_min_x or intersection_max_y <= intersection_min_y:
        return False  # No intersection
    
    intersection_area = (intersection_max_x - intersection_min_x) * (intersection_max_y - intersection_min_y)
    r1_area = (max_x1 - min_x1) * (max_y1 - min_y1)
    r2_area = (max_x2 - min_x2) * (max_y2 - min_y2)

    print('areas')
    print(r1_area)
    print(r2_area)
    print(intersection_area)
    
    return (intersection_area > threshold * r1_area and intersection_area > threshold * r2_area)

# Test the function
print(matched(0, 0, 10, 9, 2, 1, 11, 11))  # Should return True
print(matched(0, 0, 10, 10, 15, 15, 20, 20))  # Should return False
