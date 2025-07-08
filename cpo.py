from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import Point
import numpy as np


def simplify_polygon_vertices(vertices, tolerance=1e-6):
    """
    Simplify the vertex list of a polygon by removing co-linear vertices
    """

    if len(vertices) < 3:
        return vertices

    simplified_vertices = []
    n = len(vertices)

    for i in range(n):
        p1 = Point(vertices[(i - 1) % n]) 
        p2 = Point(vertices[i])         
        p3 = Point(vertices[(i + 1) % n]) 

        polygon = Polygon([p1, p2, p3])
        area = polygon.area

        if abs(area) > tolerance:
            simplified_vertices.append(vertices[i])

    if len(simplified_vertices) < 3 and len(vertices) >= 3:
        return vertices 
    
    return simplified_vertices


def judge_angle(p1, p2, p3):
    """
    Determining the convexity of a corner
    """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    vt = np.array([v1[1], -v1[0]])
    return np.sign(np.dot(vt, v2))


def rotate_polygon(polygon):
    """
    Help select the starting point of the polygon
    """
    rotate_flag = True
    while rotate_flag:
        rotate_flag = False
        p1 = polygon[-1]
        p2 = polygon[0]
        p3 = polygon[1]
        if judge_angle(p1, p2, p3) > 0:
            polygon.insert(0, polygon.pop())
            rotate_flag = True
            
    return polygon


def generate_layer(polygon):
    polygon = simplify_polygon_vertices(polygon, tolerance=1e-6)
    polygon = rotate_polygon(polygon)
    # print(len(polygon))
    ring_outer = LinearRing(polygon+[polygon[0]])
    # print(ring_outer)

    ring_outer_list = [ring_outer.parallel_offset(0.2, side='left', join_style='mitre')]

    while(1):
        ring_outer_offset = ring_outer_list[-1].parallel_offset(0.4, side='left', join_style='mitre')
        try:
            if Polygon(ring_outer_offset).area < 0.1:
                break
        except:
            break
        ring_outer_list.append(ring_outer_offset)

    point_list = []
    for ring in ring_outer_list:
        for point in ring.coords:
            point_list.append(point)
            
    return point_list