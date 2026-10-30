"""
detector.py : planche à trous + robot (ArUco)

Deux usages :
    python detector.py image image.jpg   -> matrice d'occupation sur une photo (+ images de debug)
    python detector.py live              -> position (x, y, z) du robot en temps réel (Intel RealSense D435i)

En mode live : Échap = quitter, 'o' = matrice d'occupation sur l'image courante,
'b' = recalculer la pose de la planche.

Repères :
  - image redressée de la planche (extract_board) : origine en haut à gauche, y vers le bas
  - pose 3D / position du robot : origine au coin BAS GAUCHE de la planche,
    x vers la droite, y vers le haut, z hors de la planche (vers la caméra), en mm
"""
import argparse

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constantes (en mm)
# ---------------------------------------------------------------------------
BOARD_X = 800.0          # extension en x de la planche redressée (portrait)
BOARD_Y = 1400.0         # extension en y

ARUCO_SIZE = 100.0       # côté du carré noir des tags de la planche
ARUCO_OFFSET = 18.0      # bord de la planche -> bord du tag
HOLE_SPACING = 15.0
MAX_HOLE_DIST_RATIO = 0.5   # un bloc compte pour un trou si son centre en est à < ratio x pas

SCALE = 2               # px/mm de l'image redressée (une case de 15 mm = 30 px)

ROBOT_TAG_SIZE = 80.0    # côté du CARRÉ NOIR du tag du robot : À MESURER
ROBOT_TAG_ID = None      # ex. 7 ; None = premier tag détecté hors IDs 1-4
ROBOT_YAW_OFFSET = 0.0   # degrés, pour aligner le cap du tag avec l'avant du robot

FREEZE_BOARD = True      # True : pose de la planche calculée une fois (caméra fixe)
SMOOTHING = 0.3          # lissage exponentiel de la position (1 = aucun lissage)


# ---------------------------------------------------------------------------
# Tags de la planche
# ---------------------------------------------------------------------------
def tag_corners(x, y):
    """Coins d'un tag dont le coin haut-gauche est en (x, y) : HG, HD, BD, BG."""
    s = ARUCO_SIZE
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


_far_x = BOARD_X - ARUCO_OFFSET - ARUCO_SIZE
_far_y = BOARD_Y - ARUCO_OFFSET - ARUCO_SIZE

# Coordonnées des coins dans le repère de la planche (origine haut gauche, y vers le bas)
corner_aruco = {
    1: tag_corners(ARUCO_OFFSET, ARUCO_OFFSET),   # haut-gauche
    2: tag_corners(_far_x,       ARUCO_OFFSET),   # haut-droite
    3: tag_corners(ARUCO_OFFSET, _far_y),         # bas-gauche
    4: tag_corners(_far_x,       _far_y),         # bas-droite
}


def make_detector(dict_id):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)


# Créés une seule fois (coûteux à recréer à chaque image)
BOARD_DETECTOR = make_detector(cv2.aruco.DICT_4X4_100)
ROBOT_DETECTOR = make_detector(cv2.aruco.DICT_4X4_50)


def get_aruco_corners_id(image, detector=BOARD_DETECTOR):
    """Détecte les tags et renvoie {id: coins (4, 2)} en pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = detector.detectMarkers(gray)
    positions = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            positions[int(marker_id)] = corners[i][0]
    return positions


# ---------------------------------------------------------------------------
# Redressement de la planche (homographie sur les 16 coins)
# ---------------------------------------------------------------------------
def extract_board(image, positions=None):
    if positions is None:
        positions = get_aruco_corners_id(image, BOARD_DETECTOR)

    missing = [t for t in corner_aruco if t not in positions]
    if missing:
        raise ValueError(f"Tags de la planche non détectés : {missing}")

    src, dst = [], []
    for tag_id, board_pts in corner_aruco.items():
        for img_pt, (bx, by) in zip(positions[tag_id], board_pts):
            src.append(img_pt)
            dst.append((bx * SCALE, by * SCALE))
    src = np.array(src, dtype=np.float32)
    dst = np.array(dst, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, method=0)

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1) / SCALE
    print(f"Redressement : erreur de reprojection moy {err.mean():.2f} mm, max {err.max():.2f} mm")

    return cv2.warpPerspective(image, H, (int(BOARD_X * SCALE), int(BOARD_Y * SCALE)))


# ---------------------------------------------------------------------------
# Grille de trous (sur l'image redressée)
# ---------------------------------------------------------------------------
def _odd(n):
    n = int(n)
    return n if n % 2 else n + 1


def _hole_response(image, scale):
    """Carte où les petites taches sombres (trous) ressortent."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.5 * scale)

    k = _odd(HOLE_SPACING * scale * 0.7)   # plus grand qu'un trou, plus petit qu'un bloc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    resp = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    resp = np.clip(resp - resp.mean(), 0, np.percentile(resp, 99))

    # Ignore les zones des tags ArUco (coins)
    m = int((ARUCO_OFFSET + ARUCO_SIZE + 10) * scale)
    resp[:m, :m] = 0
    resp[:m, -m:] = 0
    resp[-m:, :m] = 0
    resp[-m:, -m:] = 0
    return resp


def _fit_axis(profile, period_nominal):
    """Trouve (position du premier pic, période) d'un signal périodique 1D."""
    x = np.arange(len(profile))
    profile = profile - profile.mean()

    best = None
    for p in np.linspace(0.97 * period_nominal, 1.03 * period_nominal, 41):
        c = np.sum(profile * np.exp(-2j * np.pi * x / p))
        if best is None or abs(c) > best[0]:
            best = (abs(c), p, c)

    _, p, c = best
    x0 = (-np.angle(c) / (2 * np.pi) * p) % p
    return x0, p


def find_grid(image):
    """Positions (px) des colonnes xs et des lignes ys de trous."""
    h, w = image.shape[:2]
    scale = w / BOARD_X
    period = HOLE_SPACING * scale

    resp = _hole_response(image, scale)
    x0, px = _fit_axis(resp.sum(axis=0), period)
    y0, py = _fit_axis(resp.sum(axis=1), period)

    print(f"Grille : période colonnes {px / scale:.2f} mm, lignes {py / scale:.2f} mm "
          f"(attendu {HOLE_SPACING} mm)")
    return np.arange(x0, w, px), np.arange(y0, h, py)


# ---------------------------------------------------------------------------
# Détection des blocs
# ---------------------------------------------------------------------------
def block_mask(image, scale):
    """Masque binaire des blocs colorés (bleu, vert, orange)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 130) & (hsv[..., 2] > 90)).astype(np.uint8) * 255

    k = _odd(HOLE_SPACING * scale * 0.4)   # retire les fines structures (corde, bruit)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def block_centroids(board, mask, scale, peak_ratio=0.7):
    """Centroïdes des blocs, en séparant ceux qui se touchent (distance transform + watershed)."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Seuil relatif au maximum de CHAQUE tache (et non de toute l'image),
    # pour ne pas perdre les blocs plus petits (partiellement masqués, etc.)
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    sure_fg = np.zeros(mask.shape, dtype=np.uint8)
    for k in range(1, n_cc):
        x, y, w, h = stats[k, :4]
        inside = labels[y:y + h, x:x + w] == k
        d = dist[y:y + h, x:x + w]
        sub = sure_fg[y:y + h, x:x + w]
        sub[inside & (d > peak_ratio * d[inside].max())] = 1

    n_fg, markers = cv2.connectedComponents(sure_fg)

    # 1 = fond, 2.. = blocs, 0 = zone inconnue répartie par le watershed
    markers = markers + 1
    markers[mask == 0] = 1
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


# ---------------------------------------------------------------------------
# Matrice d'occupation
# ---------------------------------------------------------------------------
def subdivide_board(image, xs, ys, occupation=None):
    """Debug : trace les lignes/colonnes de trous et entoure les trous occupés."""
    result = image.copy()
    h, w = result.shape[:2]

    for x in xs:
        cv2.line(result, (int(round(x)), 0), (int(round(x)), h - 1), (0, 255, 0), 1)
    for y in ys:
        cv2.line(result, (0, int(round(y))), (w - 1, int(round(y))), (0, 255, 0), 1)

    if occupation is not None:
        radius = int(HOLE_SPACING * SCALE * 0.35)
        for j, i in zip(*np.nonzero(occupation)):
            cv2.circle(result, (int(round(xs[i])), int(round(ys[j]))), radius, (0, 0, 255), 2)
    return result


def create_occupation_matrix(image, positions=None, debug=False):
    """
    Matrice d'occupation sur les trous : M[j, i] = 1 si le centre d'un bloc est
    proche du trou (ligne j, colonne i), 0 sinon.
    Renvoie (occupation, xs, ys) avec xs, ys les positions des trous en pixels de l'image redressée.
    """
    board = extract_board(image, positions)
    scale = board.shape[1] / BOARD_X

    xs, ys = find_grid(board)
    mask = block_mask(board, scale)

    occupation = np.zeros((len(ys), len(xs)), dtype=np.uint8)

    for cx, cy in block_centroids(board, mask, scale):
        i = int(np.argmin(np.abs(xs - cx)))
        j = int(np.argmin(np.abs(ys - cy)))
        occupation[j-1, i-1] = 1

    print(f"Matrice {occupation.shape[0]} x {occupation.shape[1]}, {occupation.sum()} blocs")

    if debug:
        cv2.imwrite("board.jpg", board)
        cv2.imwrite("mask.jpg", mask)
        cv2.imwrite("sub_board.jpg", subdivide_board(board, xs, ys, occupation))
    return occupation, xs, ys


def save_occupation(occupation, xs, ys, path="occupation.npz"):
    """Sauvegarde la matrice et la position des trous dans le repère du robot
    (origine en bas à gauche, y vers le haut, en mm)."""
    scale = SCALE
    np.savez(path,
             occupation=occupation,
             holes_x_mm=xs / scale,
             holes_y_mm=BOARD_Y - ys / scale)
    print(f"Matrice sauvegardée dans {path}")


# ---------------------------------------------------------------------------
# Pose 3D : planche et robot (repère origine bas gauche, y vers le haut)
# ---------------------------------------------------------------------------
def board_pose(positions, K, dist, verbose=False):
    """Pose de la planche : X_cam = R @ X_planche + t. Renvoie (R, t) ou None."""
    if any(t not in positions for t in corner_aruco):
        return None

    obj, img = [], []
    for tag_id, board_pts in corner_aruco.items():
        for img_pt, (bx, by) in zip(positions[tag_id], board_pts):
            obj.append((bx, BOARD_Y - by, 0.0))     # y vers le haut
            img.append(img_pt)
    obj = np.array(obj, dtype=np.float64)
    img = np.array(img, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
    if not ok:
        return None

    if verbose:
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        print(f"Pose planche : erreur de reprojection moy {err.mean():.2f} px, max {err.max():.2f} px")

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def robot_pose(positions, K, dist, R_b, t_b):
    """Position (x, y, z) en mm dans le repère de la planche et cap en degrés, ou None."""
    if ROBOT_TAG_ID is not None:
        tags = {i: c for i, c in positions.items() if i == ROBOT_TAG_ID}
    else:
        tags = {i: c for i, c in positions.items() if i not in corner_aruco}
    if not tags:
        return None
    _, corners = next(iter(tags.items()))

    s = ROBOT_TAG_SIZE / 2
    obj = np.array([[-s,  s, 0], [ s,  s, 0],
                    [ s, -s, 0], [-s, -s, 0]], dtype=np.float64)   # HG, HD, BD, BG
    ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R_t, _ = cv2.Rodrigues(rvec)

    pos = R_b.T @ (tvec.ravel() - t_b)
    R_rel = R_b.T @ R_t
    yaw = np.degrees(np.arctan2(R_rel[1, 0], R_rel[0, 0])) + ROBOT_YAW_OFFSET
    return pos, yaw


# ---------------------------------------------------------------------------
# Caméra RealSense : les intrinsèques sont lus sur la caméra branchée
# ---------------------------------------------------------------------------
class RealSenseCamera:
    # (largeur, hauteur, fps) essayés dans l'ordre après le profil demandé
    FALLBACKS = [(1920, 1080, 30), (1280, 720, 30), (1280, 720, 15),
                 (960, 540, 30), (848, 480, 30), (640, 480, 30)]

    def __init__(self, width=1920, height=1080, fps=30):
        import pyrealsense2 as rs

        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise RuntimeError("Aucune RealSense détectée : vérifie le câble et le port USB")
        dev = devices[0]

        usb = (dev.get_info(rs.camera_info.usb_type_descriptor)
               if dev.supports(rs.camera_info.usb_type_descriptor) else "?")
        print(f"Caméra : {dev.get_info(rs.camera_info.name)} (USB {usb})")
        if usb.startswith("2"):
            print("ATTENTION : connexion USB 2.x, les hautes résolutions ne sont pas disponibles. "
                  "Utilise un câble USB 3 et un port USB 3.")

        supported = sorted({(v.width(), v.height(), v.fps())
                            for s in dev.query_sensors()
                            for p in s.get_stream_profiles()
                            if p.stream_type() == rs.stream.color and p.format() == rs.format.bgr8
                            for v in [p.as_video_stream_profile()]})
        print("Profils couleur bgr8 disponibles :", supported)

        candidates = [(width, height, fps)] + [c for c in self.FALLBACKS if c != (width, height, fps)]
        last_err = None
        for w, h, f in candidates:
            if supported and (w, h, f) not in supported:
                continue
            pipe, cfg = rs.pipeline(), rs.config()
            cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            try:
                profile = pipe.start(cfg)
            except RuntimeError as e:
                last_err = e
                continue
            self.pipe = pipe
            print(f"Profil utilisé : {w}x{h} @ {f} fps")
            break
        else:
            raise RuntimeError(f"Aucun profil couleur utilisable : {last_err}")

        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx],
                           [0, i.fy, i.ppy],
                           [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(i.coeffs, dtype=np.float64)
        print(f"Intrinsèques lus sur la caméra :\n{self.K}\nDIST = {self.dist}")

        for _ in range(15):            # laisse l'auto-exposition se stabiliser
            self.pipe.wait_for_frames()

    def read(self):
        color = self.pipe.wait_for_frames().get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def close(self):
        self.pipe.stop()


# ---------------------------------------------------------------------------
# Programmes
# ---------------------------------------------------------------------------
def run_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Impossible de charger {path}")
    occupation, xs, ys = create_occupation_matrix(image, debug=True)
    save_occupation(occupation, xs, ys)


def run_live():
    cam = RealSenseCamera()
    board = None
    pos_f = None
    n = 0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = ""

            # Pose de la planche : une fois (caméra fixe) ou à chaque image
            if board is None or not FREEZE_BOARD:
                b = board_pose(get_aruco_corners_id(gray, BOARD_DETECTOR),
                               cam.K, cam.dist, verbose=board is None)
                if b is not None:
                    board = b

            if board is None:
                text = "Planche non detectee"
            else:
                result = robot_pose(get_aruco_corners_id(gray, ROBOT_DETECTOR),
                                    cam.K, cam.dist, *board)
                if result is None:
                    pos_f = None
                    text = "Robot non detecte"
                else:
                    pos, yaw = result
                    pos_f = pos if pos_f is None else (1 - SMOOTHING) * pos_f + SMOOTHING * pos
                    x, y, z = pos_f
                    text = f"x={x:.0f} y={y:.0f} z={z:.0f} mm  cap={yaw:.0f} deg"
                    n += 1
                    if n % 10 == 0:
                        print(text)

            show = cv2.resize(frame, None, fx=0.4, fy=0.4)
            cv2.putText(show, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("camera", show)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                       # Échap
                break
            if key == ord("b"):
                board = None
            if key == ord("o"):
                try:
                    occupation, xs, ys = create_occupation_matrix(frame, debug=True)
                    save_occupation(occupation, xs, ys)
                except ValueError as e:
                    print(e)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p_img = sub.add_parser("image", help="matrice d'occupation sur une photo")
    p_img.add_argument("path", nargs="?", default="image.jpg")
    sub.add_parser("live", help="position du robot en temps réel (RealSense)")
    args = parser.parse_args()

    if args.mode == "image":
        run_image(args.path)
    else:
        run_live()
