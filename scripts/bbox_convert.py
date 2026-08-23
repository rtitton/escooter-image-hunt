"""Conversione di annotazioni YOLO da poligono a bounding box.

Una riga di annotazione YOLO poligono è
    classe x1 y1 x2 y2 ... xn yn
Il bounding box minimo che contiene esattamente il poligono è quello i cui
lati passano per i valori min/max delle coordinate dei vertici.
"""

BBOX_FIELDS = 5  # classe, x_centro, y_centro, larghezza, altezza


def convert_line(line: str) -> str:
    """Converte una riga di annotazione in bounding box; le righe già in
    formato bbox (5 campi) sono lasciate invariate."""
    parts = line.split()
    if len(parts) <= BBOX_FIELDS:
        return line

    cls = parts[0]
    coords = list(map(float, parts[1:]))
    xs, ys = coords[0::2], coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
    w, h = xmax - xmin, ymax - ymin
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def convert_label_file(text: str) -> str:
    lines = [convert_line(line) for line in text.splitlines() if line.strip()]
    return "\n".join(lines) + ("\n" if lines else "")
