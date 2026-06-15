import os
root = r"C:\Users\user\Downloads\CelebA_Spoof"
for dirpath, dirnames, filenames in os.walk(root):
    depth = dirpath.replace(root, "").count(os.sep)
    if depth <= 4:
        exts = {os.path.splitext(f)[1].lower() for f in filenames}
        print("  " * depth + os.path.basename(dirpath) + f"/  ({len(filenames)} files, {exts})")
    if depth > 4:
        dirnames[:] = []