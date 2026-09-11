"""Remove near-white background texture without globally whitening the hand."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_propagation

path = Path(__file__).resolve().parent / "hand_palm_actor_clean.png"
original = np.array(Image.open(path).convert("RGB"))
# Only near-white, neutral pixels connected to the image border are background.
candidate = (original.min(axis=2) >= 248) & (np.ptp(original, axis=2) <= 4)
seeds = np.zeros(candidate.shape, dtype=bool)
seeds[[0, -1], :] = candidate[[0, -1], :]
seeds[:, [0, -1]] = candidate[:, [0, -1]]
background = binary_propagation(seeds, mask=candidate)
cleaned = original.copy()
cleaned[background] = 255
assert np.array_equal(cleaned[~background], original[~background])
Image.fromarray(cleaned).save(path)
saved = np.array(Image.open(path).convert("RGB"))
for name, region in (("corner", saved[:60, :60]),
                     ("left margin", saved[600:800, :80]),
                     ("finger gap", saved[230:290, 550:650])):
    assert np.all(region == 255), name
    print(f"{name}: RGB(255, 255, 255)")
print(f"Background pixels: {int(background.sum())}; all other pixels unchanged.")
