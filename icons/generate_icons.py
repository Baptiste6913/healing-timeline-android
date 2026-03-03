"""Generate PWA icons as PNG files using pure Python (no Pillow needed)."""
import struct
import zlib
import os

def create_png(width, height, pixels):
    """Create a minimal PNG from raw RGBA pixel data."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    header = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))

    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter: none
        for x in range(width):
            idx = (y * width + x) * 4
            raw += bytes(pixels[idx:idx+4])

    idat = chunk(b'IDAT', zlib.compress(raw, 9))
    iend = chunk(b'IEND', b'')
    return header + ihdr + idat + iend

def draw_icon(size):
    """Draw Healing Timeline icon: gradient circle with a timeline wave."""
    pixels = [0] * (size * size * 4)
    cx, cy = size / 2, size / 2
    r_outer = size * 0.45
    r_inner = size * 0.38

    for y in range(size):
        for x in range(size):
            idx = (y * size + x) * 4
            dx = x - cx
            dy = y - cy
            dist = (dx*dx + dy*dy) ** 0.5

            if dist <= r_outer:
                # Background: dark gradient
                t = dist / r_outer
                bg_r = int(10 + t * 16)
                bg_g = int(10 + t * 16)
                bg_b = int(10 + t * 16)
                a = 255

                # Accent ring
                if dist > r_inner:
                    ring_t = (dist - r_inner) / (r_outer - r_inner)
                    # Gradient: #0077b6 -> #00b4d8
                    ring_r = int(0 * (1-ring_t) + 0 * ring_t)
                    ring_g = int(119 * (1-ring_t) + 180 * ring_t)
                    ring_b = int(182 * (1-ring_t) + 216 * ring_t)
                    mix = 0.9
                    bg_r = int(bg_r * (1-mix) + ring_r * mix)
                    bg_g = int(bg_g * (1-mix) + ring_g * mix)
                    bg_b = int(bg_b * (1-mix) + ring_b * mix)

                # Timeline wave in center
                norm_x = (x - cx) / (r_inner * 0.8)
                if abs(norm_x) < 1:
                    import math
                    wave_y = cy - math.sin(norm_x * math.pi * 2) * size * 0.08
                    wave_y2 = cy + size * 0.05 - math.exp(-((norm_x+0.3)**2)*4) * size * 0.12

                    # Swelling line (top)
                    if abs(y - wave_y) < size * 0.015:
                        bg_r, bg_g, bg_b = 0, 180, 216  # accent color
                    # Bruising line (bottom)
                    if abs(y - wave_y2) < size * 0.012:
                        bg_r, bg_g, bg_b = 128, 80, 180  # purple

                # Chart icon text area (center dot)
                if dist < size * 0.04:
                    bg_r, bg_g, bg_b = 0, 180, 216

                pixels[idx] = min(255, max(0, bg_r))
                pixels[idx+1] = min(255, max(0, bg_g))
                pixels[idx+2] = min(255, max(0, bg_b))
                pixels[idx+3] = a
            else:
                # Anti-alias edge
                edge = dist - r_outer
                if edge < 1.5:
                    alpha = int(255 * max(0, 1 - edge / 1.5))
                    pixels[idx] = 10
                    pixels[idx+1] = 10
                    pixels[idx+2] = 10
                    pixels[idx+3] = alpha
                else:
                    pixels[idx+3] = 0

    return pixels

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for size in [192, 512]:
        print(f'Generating {size}x{size} icon...')
        px = draw_icon(size)
        png = create_png(size, size, px)
        path = os.path.join(script_dir, f'icon-{size}.png')
        with open(path, 'wb') as f:
            f.write(png)
        print(f'  -> {path} ({len(png)} bytes)')
    print('Done!')
