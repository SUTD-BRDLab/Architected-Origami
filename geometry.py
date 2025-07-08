import numpy as np


def calculate_arc(theta, Delta=0.8, mid_layer_count = 10, layer_height = 0.4):
    """
    Calculate the arc length of a polygon edge for each layer

    Args:
        theta (float): bending angle of the edge

    Returns:
        dl (list, float): list of move-inside-distance for each layer
    """
    theta = np.deg2rad(theta)
    a = mid_layer_count * layer_height
    h = 2 * layer_height
    delta = layer_height
    r = Delta / theta
    b = a * np.tan(theta/2) + 1/np.cos(theta/2)*(h - (r - delta/2) * np.sin(theta/2))
    
    dl = []
    for i in range (mid_layer_count):
        x = np.sqrt(a ** 2 - (a - layer_height * (1/2 + i)) ** 2)
        dl.append(x*b/a)

    # for i in range (len(dl)):
    #     dl[i] -= Delta/2
        
    return dl


def calculate_layer_polygon(polygon, layer_index, mid_layer_count, top=False):
    
    if top:
        layer_index = mid_layer_count - layer_index - 1
    
    pp = []
    for v in polygon[0]:
        pp.append(v)
        
    for j in range (len(polygon[1])):
        v_index = polygon[1][j][0]
        
        if polygon[1][j][1] == 0:
            d = calculate_arc(polygon[1][j][2])[mid_layer_count - layer_index - 1]
        else:
            d = calculate_arc(polygon[1][j][2])[layer_index]    
        
        dv = np.array(polygon[0][(v_index)%len(polygon[0])])- np.array(polygon[0][v_index-1])
        dv = [-dv[1], dv[0]]
        dv = np.array(dv)/np.linalg.norm(dv)
        pp[v_index] = (np.array(pp[v_index]) + dv*d).tolist()
        pp[v_index-1] = (np.array(pp[v_index-1]) + dv*d).tolist()
    return pp


def build_prism_bottom(polygon_base, polygons_middle, mid_layer_count):
    
    prism_base = []
    prisms_middle = []
    
    has_bottom = False
    has_top = False
    
    for edge in polygon_base[1]:
        if edge[1] == 1:
            has_bottom = True
        elif edge[1] == 0:
            has_top = True
            
    if len(polygons_middle) == 1:
        if has_bottom:
            if has_top:
                layer_count = mid_layer_count / 2
            else:
                layer_count = mid_layer_count
        else:
            layer_count = 0
            
    else:
        if has_bottom:
            if has_top:
                layer_count = mid_layer_count / 2
            else:
                layer_count = mid_layer_count / 2 + 2
        else:
            layer_count = mid_layer_count / 2 - 2
            
    # print('layer_count', layer_count)
                            
    if len(polygons_middle) == 1:
        for layer_index in range (int(layer_count)):
            pp = calculate_layer_polygon(polygon_base, layer_index, mid_layer_count)
            prism_base.append(pp)
            
    else:
        for layer_index in range (int(mid_layer_count / 2 - 2)):
            pp = calculate_layer_polygon(polygon_base, layer_index, mid_layer_count)
            prism_base.append(pp)
        
        for i in range(len(polygons_middle)):
            # print('middle polygon ', i)
            prisms_middle.append([])
            polygon = polygons_middle[i]
            for layer_index in range (int(mid_layer_count / 2 - 2), int(layer_count)):
                pp = calculate_layer_polygon(polygon, layer_index, mid_layer_count)
                prisms_middle[i].append(pp)
                # print('layer_index', layer_index, pp)
                
    return prism_base, prisms_middle

        
    # run_flag = False
    # for edge in polygon[1]:
    #     if edge[1] == 1:
    #         run_flag = True
    #         break    
        
    # if run_flag:    
    #     for i in range (int(layer_count)):
    #         pp = []
    #         for v in polygon[0]:
    #             pp.append(v)
    #         for j in range (len(polygon[1])):
    #             v_index = polygon[1][j][0]
                
    #             if polygon[1][j][1] == 0:
    #                 d = calculate_arc(polygon[1][j][2])[mid_layer_count - i - 1]
    #             else:
    #                 d = calculate_arc(polygon[1][j][2])[i]    
                
    #             dv = np.array(polygon[0][(v_index)%len(polygon[0])])- np.array(polygon[0][v_index-1])
    #             dv = [-dv[1], dv[0]]
    #             dv = np.array(dv)/np.linalg.norm(dv)
    #             pp[v_index] = (np.array(pp[v_index]) + dv*d).tolist()
    #             pp[v_index-1] = (np.array(pp[v_index-1]) + dv*d).tolist()
    #         prism.append(pp)
        
    # return prism


def build_prism_top(polygon_base, polygons_middle, mid_layer_count):
    
    prism_base = []
    prisms_middle = []
    
    has_bottom = False
    has_top = False
    
    for edge in polygon_base[1]:
        if edge[1] == 0:
            has_bottom = True
        elif edge[1] == 1:
            has_top = True
            
    if len(polygons_middle) == 1:
        if has_bottom:
            if has_top:
                layer_count = mid_layer_count / 2
            else:
                layer_count = mid_layer_count
        else:
            layer_count = 0
            
    else:
        if has_bottom:
            if has_top:
                layer_count = mid_layer_count / 2
            else:
                layer_count = mid_layer_count / 2 + 2
        else:
            layer_count = mid_layer_count / 2 - 2
            
    # print('layer_count', layer_count)
                            
    if len(polygons_middle) == 1:
        for layer_index in range (int(layer_count)):
            pp = calculate_layer_polygon(polygon_base, layer_index, mid_layer_count, top=True)
            prism_base.append(pp)
            
    else:
        for layer_index in range (int(mid_layer_count / 2 - 2)):
            pp = calculate_layer_polygon(polygon_base, layer_index, mid_layer_count, top=True)
            prism_base.append(pp)
        
        for i in range(len(polygons_middle)):
            # print('middle polygon ', i)
            prisms_middle.append([])
            polygon = polygons_middle[i]
            for layer_index in range (int(mid_layer_count / 2 - 2), int(layer_count)):
                pp = calculate_layer_polygon(polygon, layer_index, mid_layer_count, top=True)
                prisms_middle[i].append(pp)
                # print('layer_index', layer_index, pp)
                
    return prism_base, prisms_middle
    
    # prism = []
    
    # layer_count = mid_layer_count
    # for edge in polygon[1]:
    #     if edge[1] == 1:
    #         layer_count /= 2
    #         break
        
    # run_flag = False
    # for edge in polygon[1]:
    #     if edge[1] == 0:
    #         run_flag = True
    #         break    
        
    # if run_flag:           
    #     for i in range (int(layer_count)):
    #         pp = []
    #         for v in polygon[0]:
    #             pp.append(v)
    #         for j in range (len(polygon[1])):
    #             v_index = polygon[1][j][0]
                
    #             if polygon[1][j][1] == 1:
    #                 d = calculate_arc(polygon[1][j][2])[mid_layer_count - i - 1]
    #             else:
    #                 d = calculate_arc(polygon[1][j][2])[i]    
                
    #             dv = np.array(polygon[0][(v_index)%len(polygon[0])])- np.array(polygon[0][v_index-1])
    #             dv = [-dv[1], dv[0]]
    #             dv = np.array(dv)/np.linalg.norm(dv)
    #             pp[v_index] = (np.array(pp[v_index]) + dv*d).tolist()
    #             pp[v_index-1] = (np.array(pp[v_index-1]) + dv*d).tolist()
    #         prism.append(pp)
            
    # return prism


def build_prism(polygon_base, polygons_middle, position, mid_layer_count=10):
    """
    Generate the prism of a polygon acording to the edge information
    """
    if position == 'bottom':
        prism_base, prisms_middle = build_prism_bottom(polygon_base, polygons_middle, mid_layer_count)
    elif position == 'top':
        prism_base, prisms_middle = build_prism_top(polygon_base, polygons_middle, mid_layer_count)
    return prism_base, prisms_middle