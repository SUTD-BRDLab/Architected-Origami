from geometry import build_prism
from cpo import generate_layer
from polygon_manipulator import generate_surface_polygons
import numpy as np


def write_header_lines(file, layer_height, line_width, layer_count, mesh, start_x, start_y, start_z=0.3):
    """
    Write the header lines of the gcode file
    """
    with open(file, 'w') as f:
        header_lines = [
            f";layer_height = {layer_height}",
            f"\n;line_width = {line_width}",
            f"\n;layer_count = {layer_count}",
            f"\n;mesh = {mesh}"
        ]
        
        initialize_lines = [
            "\nG21 ;start of the code",
            "\nG1 Z15 F300",
            "\nG28 X0 Y0 ;Home",
            "\nG92 X0 Y0 ;Consider this as current",
            "\nG0 X50 Y50 F3000 ;Go-to Offset",  
            "\nG92 X0 Y0 ;Reset",
            "\n",
            f"\nG0 F3600 X{start_x:.3f} Y{start_y:.3f} Z{start_z:.3f} ;Go to start position",
            "\nM7",
            "\nG4 P150",
            "\n\n"
        ]
        
        f.writelines(header_lines)
        f.writelines(initialize_lines)
        

def write_finish_lines(file):
    """
    Write the finish lines of the gcode file
    """
    with open(file, 'a') as f:
        finish_lines = [
            "\n\n;Finish",
            "\nM9",
            "\nG1 Z10.000"
            "\nG28 X-100 Y-100;Home",
        ]
        f.writelines(finish_lines)
        

def initialize_layer(xy, z, lines):
    """
    Generate the initial lines of a layer
    """
    # lines.append("\nG1 Z10.000")
    # lines.append(f"\nG1 X{xy[0]:.3f} Y{xy[1]:.3f}")
    lines.append(f"\nG1 X{xy[0]:.3f} Y{xy[1]:.3f} Z{z:.3f}")
    return lines


def write_move_to(file, x, y, z):
    """
    Write the move to command to the gcode file
    """
    lines = [
        f"\nM9",
        f"\nG1 Z10.000",
        f"\nG1 X{x:.3f} Y{y:.3f}",
        f"\nG1 Z{z:.3f}",
        f"\nM7",
        f"\nG4 P150"
    ]
    with open(file, 'a') as f:
        f.writelines(lines)
        

def write_layer(file, point_list, z, direction):
    """
    Draw single platform layer by concentrate infill

    Args:
        point_list (ndarray)  : np.array([[x1, y1], [x2, y2], ...])
        w (float)             : line width
        direction (int)       : 0 for out -> in (counter clockwise) 
                                1 for in -> out (clockwise)
    """
    
    if direction == 0:
        pl = point_list
    elif direction == 1:
        pl = []
        for point in point_list:
            pl.insert(0, point)
            
    lines = initialize_layer(pl[0], z, [])
    
    for p in pl:
        lines.append(f"\nG1 X{p[0]:.3f} Y{p[1]:.3f}")
        
    with open(file, 'a') as f:
        f.writelines(lines)
        
        
def go_to_next_layer(z, layer_height):
    """
    Move to the next layer
    """
    z += layer_height
    
    
def remove_duplicate_vertices(polygon, tolerance=1e-6):
    """
    删除多边形中的重复顶点。

    Args:
        polygon: 多边形顶点坐标的numpy数组，形状为 (n, 2)。
        tolerance: 用于判断两个顶点是否相同的容差值。

    Returns:
        删除重复顶点后的多边形顶点坐标的numpy数组。
    """

    polygon = np.array(polygon)
    if len(polygon) <= 1:  # 如果只有一个点或者没有点，则不需要处理
        return polygon

    simplified_polygon = [polygon[0]]  # 初始化结果列表，并将第一个顶点添加进去

    for i in range(1, len(polygon)):
        current_vertex = polygon[i]
        previous_vertex = simplified_polygon[-1]  # 上一个保留的顶点

        # 使用 np.allclose() 函数比较两个顶点是否接近相同
        if not np.allclose(current_vertex, previous_vertex, atol=tolerance):
            simplified_polygon.append(current_vertex)  # 如果不相同，则保留当前顶点

    return np.array(simplified_polygon)
    
    
def generate_gcode(file, filename, polygons_final, outer_countour, middle_polygons, direction, layer_height, line_width):
    """
    Generate the gcode file
    """
    
    for i in range(len(polygons_final)):
        if len(polygons_final[i]) == 2:
            polygons_final[i].append([]) 
    
    write_header_lines(file, layer_height, line_width, 1, f'{filename}_{direction}', 0, 0)
    polygon_base = generate_surface_polygons(polygons_final, direction)
    for polygon in polygon_base:
        write_move_to(file, polygon[0][0], polygon[0][1], 0.3)
        write_layer(file, generate_layer(polygon), 0.3, 1)

    write_move_to(file, outer_countour[0][0], outer_countour[0][1], 0.7)
    write_layer(file, generate_layer(outer_countour), 0.7, 1)

    for i in range(len(polygons_final)):
    # for i in range (1,2):
        polygon_base = polygons_final[i]
        polygons_middle = middle_polygons[i]
        
        prism_base, prisms_middle = build_prism(polygon_base, polygons_middle, direction)
        
        for layer_index, p in enumerate(prism_base):
            p = remove_duplicate_vertices(p)
            if layer_index == 0:
                write_move_to(file, p[0][0], p[0][1], 1.1)
            write_layer(file, generate_layer(p), 1.1+layer_index*0.4, layer_index%2)

        for prism_middle in prisms_middle:
            for layer_index, p in enumerate(prism_middle):
                p = remove_duplicate_vertices(p)
                if layer_index == 0:
                    write_move_to(file, p[0][0], p[0][1], 1.1+1.2+layer_index*0.4)
                write_layer(file, generate_layer(p), 1.1+1.2+layer_index*0.4, layer_index%2)
        
    # for polygon in polygons_final:
    #     prism = build_prism(polygon, direction)

    #     for layer_index, p in enumerate(prism):
    #         if layer_index == 0:
    #             write_move_to(file, p[0][0], p[0][1], 1.1)
    #         write_layer(file, generate_layer(p), 1.1+layer_index*0.4, layer_index%2)
            
    write_finish_lines(file)