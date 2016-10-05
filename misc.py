from pathlib import Path

def get_all_fits_in_tree(root):
    list(Path(root).rglob('*.fits'))