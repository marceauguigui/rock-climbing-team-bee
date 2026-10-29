"""
Prend un array binaire avec les coordonnes des prises ou non,
verifie en fonction de la position du robot et de ses dimensions quels prises sont accessibles
Each node of the graph is a coordinate with a socket

Chaque etat est un jeu de 5 couple de coordonnées 1 pour le barycentre et 1 pour chaque membre 

3 Phases de verification : 
    Prise accessible pour les 4 membres 
    Est-ce que l'etat est stable on risque pas de tomber (peut etre ne sert pas )
    Est-ce que le mouvement fait des collisions 

Plannification :
    algo de planification 
    optimization heuristique par rapport au : temps, energie, stabilité 
"""
import numpy as np
OVERSHOOT = 14 #en mm, c'est le rayon de la sphere des membre et la distance de plus verticalement qu'on doit pouvoir atteindre sur une prise 
MIN_REACH = 40 #en mm estimation, a voir ce que le robot peut faire et aussi si on prend en compte qu'il peut minimiser cette valeur en augmentant la distance entre lui et le mur (probleme plan ou pas)
MAX_REACH = 00
TORSO_WIDTH = 40 #en mm
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# State represented by 2D coordinates (y, x) for 4 limbs + torso CoM
@dataclass(frozen=True)
class StanceState:
    lh: Tuple[int, int]  # Left Hand
    rh: Tuple[int, int]  # Right Hand
    lf: Tuple[int, int]  # Left Foot
    rf: Tuple[int, int]  # Right Foot
    com: Tuple[float, float]  # Center of Mass (y, x)

def get_reachable_sockets(
    limb_pos: Tuple[int, int], 
    grid: np.ndarray, 
    max_reach: float
) -> List[Tuple[int, int]]:
    """
    Finds all active sockets (1s) within reach radius.
    Returns a dynamic list of varying length.
    """
    ly, lx = limb_pos
    h, w = grid.shape
    
    # Define bounding box slice to avoid scanning the entire grid
    y_min, y_max = max(0, int(ly - max_reach)), min(h, int(ly + max_reach + 1))
    x_min, x_max = max(0, int(lx - max_reach)), min(w, int(lx + max_reach + 1))
    
    sub_grid = grid[y_min:y_max, x_min:x_max]
    socket_y, socket_x = np.where(sub_grid == 1)
    
    # Convert local slice coordinates back to global wall coordinates
    global_sockets = []
    for sy, sx in zip(socket_y + y_min, socket_x + x_min):
        dist = np.hypot(sy - ly, sx - lx)
        if 0 < dist <= max_reach:  # Ignore current position
            global_sockets.append((int(sy), int(sx)))
            
    return global_sockets  # <--- Dynamically sized list (e.g., 2, 8, or 0 items)

def heuristic(barycentre_pos,scaling=1):
    """
    barycentre_pos est souvent le 5eme element de l'état du robot 
    la distance jusqu'au socket le plus superieur, lateralement ce n'est pas trop grave 
    """
    return scaling*barycentre_pos[0] # a adapter dependant de la position des axes 

def heuristic_stabilite(etat_robot, scaling=1):
    """
    etat_robot[0] est la jambe droite ou gauche 
    etat_robot[1] est la jambe droite ou gauche 
    etat_robot[2] est le bras droite ou gauche 
    etat_robot[3] est le bras droite ou gauche 
    On veut eviter de ne pas avoir les membre trop ecartes 
    probablement trop long pour etre necessaire 
    """
    distance_jambes = np.abs(etat_robot[0][1] - etat_robot[1][1])
    distance_bras =  scaling* np.abs(etat_robot[2][1] - etat_robot[3][1])
    
    return np.max(TORSO_WIDTH, distance_bras,distance_jambes )
 
def prise_accecible(membre_robot_pos:np.ndarray, pos_barycentre, pos_cible:np.ndarray)-> float :
    """
    Dit si une prise est accessible par rapport au barycentre du robot 1 
    si non retourne 0. Donne un facteur de la longeur du bras de combien on peut atteindre plus haut verticallement 
    Si on ne peut pas y acceder mais presque ca renvoie un facteur entre 0 et 1 exclu de la distance du bras pour qu'on puisse l'atteindre 
    Attention : Prendre en compte que on doit overshoot un peu la position de la cible pour dire si elle est acessible 
    Args:
        pos_robot: 
        pos_cible: 

    Returns:
        a quel point on est proche 

    """
    #doit prendre en compte le OVERSHOOT le MIN_REACH et MAX_REACH

    return np.array([1,0,0,0.2])


def reachable_sockets(etat_robot):
    possible_sockets = np.zeros()
    # possible sockets needt to check for support_polygon stability 
    for socket in possible_sockets:
        for membre in etat_robot[:-1]:
            prise_accecible(membre,etat_robot[-1],socket)
    # return un array avec pour chaque membre les coordonnes de socket possible 
    # idealement cela devrait renvoyer les plages position que le barycentre peut prendre en gardant les 4 membres immobiles 
#TODO comment gerer les mouvement de barycentre quand on a les quatre membre pris ?





def is_point_in_triangle(
    p: Tuple[float, float], 
    a: Tuple[float, float], 
    b: Tuple[float, float], 
    c: Tuple[float, float],
    margin: float = 0.5
) -> bool:
    """
    The 2D Center of Mass projection (COM) must lie inside the support triangle
    formed by the 3 remaining planted limbs during a movement.
    Checks if 2D Point `p` lies inside triangle (a, b, c) using Cross Products.
    
    Parameters:
        p: Target point (CoM) -> (y, x)
        a, b, c: Triangle vertices (grounded limb sockets) -> (y, x)
        margin: Distance buffer to keep CoM away from triangle edges (in grid units)
        
    Returns:
        bool: True if CoM is safely inside the support triangle.
    """
    py, px = p
    ay, ax = a
    by, bx = b
    cy, cx = c

    # Helper function for 2D cross product sign
    def cross_product(p1, p2, p3):
        return (p1[1] - p3[1]) * (p2[0] - p3[0]) - (p2[1] - p3[1]) * (p1[0] - p3[0])

    # Compute cross products for each triangle edge
    d1 = cross_product(p, a, b)
    d2 = cross_product(p, b, c)
    d3 = cross_product(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    # Point is inside if all cross products share the same sign
    inside = not (has_neg and has_pos)
    if not inside:
        return False

    # Check margin distance to all three triangle edge segments
    vertices = [np.array(a), np.array(b), np.array(c)]
    p_arr = np.array(p)

    for i in range(3):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 3]
        
        # Calculate distance from point p_arr to line segment (v1, v2)
        edge = v2 - v1
        edge_len_sq = np.dot(edge, edge)
        if edge_len_sq == 0:
            dist = np.linalg.norm(p_arr - v1)
        else:
            t = max(0, min(1, np.dot(p_arr - v1, edge) / edge_len_sq))
            projection = v1 + t * edge
            dist = np.linalg.norm(p_arr - projection)
            
        if dist < margin:
            return False  # Too close to the support edge

    return True


def check_support_polygon(
    swing_limb: str, 
    current_stance: StanceState, 
    stability_margin: float = 0.5
) -> bool:
    """
    Validates if the robot remains balanced when `swing_limb` lifts off the wall.
    """
    # 1. Collect coordinates of the 3 grounded limbs
    grounded_limbs: List[Tuple[int, int]] = [
        pos for name, pos in current_stance.limbs.items() if name != swing_limb
    ]
    
    a, b, c = grounded_limbs[0], grounded_limbs[1], grounded_limbs[2]
    
    # 2. Verify CoM lies inside the support triangle formed by the 3 anchored limbs
    return is_point_in_triangle(
        p=current_stance.com, 
        a=a, 
        b=b, 
        c=c, 
        margin=stability_margin
    )



def generate_climbing_grid(
    width: int = 52, 
    height: int = 93, 
    density: float = 0.08, 
    seed: int = 42
) -> np.ndarray:
    """
    TODO have to addapt so that two neigbouring points dont both have 1's as a 1 signifies the center and that it occupies the 8 surounding points 
    Generates a 2D climbing wall matrix.
    
    Parameters:
        width  (int)  : Number of horizontal grid units (X axis -> 52)
        height (int)  : Number of vertical grid units (Y axis -> 93)
        density (float): Probability [0, 1] of a socket appearing at a cell
        seed   (int)  : Random seed for reproducibility
        
    Returns:
        np.ndarray: Matrix of shape (height, width) with 1s (sockets) and 0s (empty space).
    """
    np.random.seed(seed)
    
    # Generate random binary grid based on density probability
    grid = (np.random.rand(height, width) < density).astype(np.int8)
    
    # Ensure starting row (bottom) and target row (top) have sufficient sockets
    grid[-1, :] = (np.random.rand(width) < 0.30).astype(np.int8)  # Start holds
    grid[0, :] = (np.random.rand(width) < 0.30).astype(np.int8)   # Goal holds
    
    return grid