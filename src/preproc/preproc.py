import cv2
import numpy as np

# Repère du board redressé (mm), origine en haut à gauche, x vers la droite, y vers le bas
BOARD_X = 800.0     # extension en x (ex-BOARD_HEIGHT)
BOARD_Y = 1400.0    # extension en y (ex-BOARD_WIDTH)

ARUCO_SIZE = 100.0
ARUCO_OFFSET = 18.0
HOLE_SPACING = 15.0
SCALE = 4           # px/mm -> une case de 15 mm = 60 px

def tag_corners(x, y):
    """Coins d'un tag dont le coin haut-gauche est en (x, y) : HG, HD, BD, BG."""
    s = ARUCO_SIZE
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]

far_x = BOARD_X - ARUCO_OFFSET - ARUCO_SIZE
far_y = BOARD_Y - ARUCO_OFFSET - ARUCO_SIZE

corner_aruco = {
    1: tag_corners(ARUCO_OFFSET, ARUCO_OFFSET),  # haut-gauche
    2: tag_corners(far_x,        ARUCO_OFFSET),  # haut-droite
    3: tag_corners(ARUCO_OFFSET, far_y),         # bas-gauche
    4: tag_corners(far_x,        far_y),         # bas-droite
}

def get_aruco_corners_id(image):
    '''
    Detects aruco tags and returns their position on the image.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)
    positions = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            positions[marker_id] = corners[i][0]
    return positions

def extract_board(image):
    positions = get_aruco_corners_id(image)   # ta fonction, inchangée

    missing = [t for t in corner_aruco if t not in positions]
    if missing:
        raise ValueError(f"Tags non détectés : {missing}")

    # 16 correspondances : 4 tags x 4 coins
    src, dst = [], []
    for tag_id, board_pts in corner_aruco.items():
        for img_pt, (bx, by) in zip(positions[tag_id], board_pts):
            src.append(img_pt)
            dst.append((bx * SCALE, by * SCALE))
    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=0)

    # Contrôle de cohérence
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1) / SCALE
    print(f"Erreur de reprojection : moy {err.mean():.2f} mm, max {err.max():.2f} mm")

    board = cv2.warpPerspective(image, H, (int(BOARD_X * SCALE), int(BOARD_Y * SCALE)))
    cv2.imwrite("board.jpg", board)
    return board

def _odd(n):
    n = int(n)
    return n if n % 2 else n + 1


def _hole_response(image, scale):
    """Carte où les petites taches sombres (trous) ressortent."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.5 * scale)

    k = _odd(HOLE_SPACING * scale * 0.7)          # plus grand qu'un trou, plus petit qu'un bloc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    resp = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    # Limite l'influence des pixels très forts, garde seulement le positif
    resp = np.clip(resp - resp.mean(), 0, np.percentile(resp, 99))

    # Ignore les zones des tags ArUco (coins)
    m = int((ARUCO_OFFSET + ARUCO_SIZE + 10) * scale)
    resp[:m, :m] = 0
    resp[:m, -m:] = 0
    resp[-m:, :m] = 0
    resp[-m:, -m:] = 0
    return resp


def _fit_axis(profile, period_nominal):
    """Trouve (premier pic, période) d'un signal périodique 1D."""
    n = len(profile)
    x = np.arange(n)
    profile = profile - profile.mean()

    best = None
    for p in np.linspace(0.97 * period_nominal, 1.03 * period_nominal, 121):
        c = np.sum(profile * np.exp(-2j * np.pi * x / p))
        if best is None or abs(c) > best[0]:
            best = (abs(c), p, c)

    _, p, c = best
    x0 = (-np.angle(c) / (2 * np.pi) * p) % p
    return x0, p


def find_grid(image):
    """Retourne les positions (en px) des colonnes xs et des lignes ys de trous."""
    h, w = image.shape[:2]
    scale = w / BOARD_X                       # px/mm déduit de l'image
    period = HOLE_SPACING * scale

    resp = _hole_response(image, scale)
    x0, px = _fit_axis(resp.sum(axis=0), period)
    y0, py = _fit_axis(resp.sum(axis=1), period)

    print(f"Période colonnes : {px / scale:.2f} mm, lignes : {py / scale:.2f} mm "
          f"(attendu {HOLE_SPACING} mm)")

    xs = np.arange(x0, w, px)
    ys = np.arange(y0, h, py)
    return xs, ys


def block_mask(image, scale):
    """Masque binaire des blocs colorés (bleu, vert, orange)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 130) & (hsv[..., 2] > 90)).astype(np.uint8) * 255

    # Retire les fines structures (rayures de la corde, bruit)
    k = _odd(HOLE_SPACING * scale * 0.4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def block_centroids(board, mask, scale, peak_ratio=0.6):
    """Centroïdes des blocs, en séparant ceux qui se touchent dans le masque."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Sommets : le coeur de chaque bloc (distance > fraction du max global)
    sure_fg = (dist > peak_ratio * dist.max()).astype(np.uint8)
    n_fg, markers = cv2.connectedComponents(sure_fg)

    # Marqueurs : 1 = fond, 2.. = blocs, 0 = zone inconnue à répartir
    markers = markers + 1
    markers[(mask == 0)] = 1
    markers[(mask > 0) & (sure_fg == 0)] = 0
    markers = cv2.watershed(board, markers.astype(np.int32))

    min_area = 0.5 * (HOLE_SPACING * scale) ** 2
    centroids = []
    for label in range(2, n_fg + 1):
        ys_, xs_ = np.nonzero(markers == label)
        if len(xs_) < min_area:
            continue
        centroids.append((xs_.mean(), ys_.mean()))
    return centroids

def subdivide_board(image, xs=None, ys=None, occupation=None):
    """
    Debug : trace les lignes/colonnes de trous, et entoure les trous occupés.
    """
    result = image.copy()
    h, w = result.shape[:2]
    if xs is None or ys is None:
        xs, ys = find_grid(image)

    for x in xs:
        cv2.line(result, (int(round(x)), 0), (int(round(x)), h - 1), (0, 255, 0), 1)
    for y in ys:
        cv2.line(result, (0, int(round(y))), (w - 1, int(round(y))), (0, 255, 0), 1)

    if occupation is not None:
        for j, i in zip(*np.nonzero(occupation)):
            cv2.circle(result, (int(round(xs[i])), int(round(ys[j]))),
                       int(HOLE_SPACING * SCALE * 0.35), (0, 0, 255), 2)
    return result

def create_occupation_matrix(image):
    board = extract_board(image)
    _, w = board.shape[:2]
    scale = w / BOARD_X

    xs, ys = find_grid(board)
    mask = block_mask(board, scale)

    occupation = np.zeros((len(ys), len(xs)), dtype=np.uint8)

    for cx, cy in block_centroids(board, mask, scale):
        i = int(np.argmin(np.abs(xs - cx)))
        j = int(np.argmin(np.abs(ys - cy)))

        occupation[j, i] = 1

    print(f"Matrice {occupation.shape[0]} x {occupation.shape[1]}, "
          f"{occupation.sum()} blocs")
    cv2.imwrite("mask.jpg", mask)
    cv2.imwrite("sub_board.jpg", subdivide_board(board, xs, ys, occupation))
    return occupation

if __name__ == "__main__":
    image = cv2.imread("image4.jpg")

    if image is None:
        raise FileNotFoundError("Impossible de charger image.jpg")

    print(create_occupation_matrix(image))
