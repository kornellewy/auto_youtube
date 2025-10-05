# transitions.py  (MoviePy ≥ 2.0)
import numpy as np
import cv2
import moviepy as mp


# ------------------------------------------------------------------
# 1.  DIFFUSION  (noisy blur ↔ sharp)
# ------------------------------------------------------------------
class DiffusionIn:
    def __init__(self, duration=2, blur_max=25, noise_max=30):
        self.d, self.b_max, self.n_max = duration, blur_max, noise_max

    def apply(self, clip):
        def f(get_frame, t):
            if t >= self.d:
                return get_frame(t)
            p = t / self.d
            frame = get_frame(t)
            b = int((1 - p) * self.b_max)
            n = (1 - p) * self.n_max
            if b:
                frame = cv2.GaussianBlur(frame, (b | 1, b | 1), 0)
            if n:
                noise = np.random.randint(-n, n + 1, frame.shape, dtype=np.int16)
                frame = cv2.add(frame, noise, dtype=cv2.CV_8U)
            return frame

        return clip.transform(f)


class DiffusionOut:
    def __init__(self, duration=2, blur_max=25, noise_max=30):
        self.d, self.b_max, self.n_max = duration, blur_max, noise_max

    def apply(self, clip):
        def f(get_frame, t):
            t0 = clip.duration - self.d
            if t < t0:
                return get_frame(t)
            p = (t - t0) / self.d
            frame = get_frame(t)
            b = int(p * self.b_max)
            n = p * self.n_max
            if b:
                frame = cv2.GaussianBlur(frame, (b | 1, b | 1), 0)
            if n:
                noise = np.random.randint(-n, n + 1, frame.shape, dtype=np.int16)
                frame = cv2.add(frame, noise, dtype=cv2.CV_8U)
            # fade to black while blurry
            fade = int((1 - p) * 255)
            return cv2.addWeighted(frame, fade / 255, np.zeros_like(frame), 0, 0)

        return clip.transform(f)


# ------------------------------------------------------------------
# 2.  GLITCH RGB SHIFT
# ------------------------------------------------------------------
class GlitchRGBIn:
    def __init__(self, duration=1, max_shift=15):
        self.d, self.s = duration, max_shift

    def apply(self, clip):
        def f(get_frame, t):
            if t >= self.d:
                return get_frame(t)
            frame = get_frame(t).astype(np.float32)
            shift = int((t / self.d) * self.s)
            b, g, r = cv2.split(frame)
            b = np.roll(b, -shift, axis=1)
            r = np.roll(r, shift, axis=1)
            return cv2.merge((b, g, r)).astype(np.uint8)

        return clip.transform(f)


class GlitchRGBOut:
    def __init__(self, duration=1, max_shift=15):
        self.d, self.s = duration, max_shift

    def apply(self, clip):
        def f(get_frame, t):
            t0 = clip.duration - self.d
            if t < t0:
                return get_frame(t)
            p = (t - t0) / self.d
            frame = get_frame(t).astype(np.float32)
            shift = int((1 - p) * self.s)
            b, g, r = cv2.split(frame)
            b = np.roll(b, -shift, axis=1)
            r = np.roll(r, shift, axis=1)
            return cv2.merge((b, g, r)).astype(np.uint8)

        return clip.transform(f)


# ------------------------------------------------------------------
# 3.  RADIAL WIPE (clock sweep)
# ------------------------------------------------------------------
class RadialWipeIn:
    def __init__(self, duration=1.5):
        self.d = duration

    def apply(self, clip):
        h, w = clip.size
        c = (w // 2, h // 2)

        def f(get_frame, t):
            if t >= self.d:
                return get_frame(t)
            angle = (t / self.d) * 360
            frame = get_frame(t)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, c, (w, h), 0, -90, angle - 90, 255, -1)
            return cv2.bitwise_and(frame, frame, mask=mask)

        return clip.transform(f)


class RadialWipeOut:
    def __init__(self, duration=1.5):
        self.d = duration

    def apply(self, clip):
        h, w = clip.size
        c = (w // 2, h // 2)

        def f(get_frame, t):
            t0 = clip.duration - self.d
            if t < t0:
                return get_frame(t)
            angle = (1 - (t - t0) / self.d) * 360
            frame = get_frame(t)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, c, (w, h), 0, -90, angle - 90, 255, -1)
            return cv2.bitwise_and(frame, frame, mask=mask)

        return clip.transform(f)


# ------------------------------------------------------------------
# 4.  PIXELATE
# ------------------------------------------------------------------
class PixelateIn:
    def __init__(self, duration=1.5, min_blocks=4):
        self.d, self.min = duration, min_blocks

    def apply(self, clip):
        w, h = clip.size

        def f(get_frame, t):
            if t >= self.d:
                return get_frame(t)
            p = t / self.d
            blocks = int(w // (self.min + (1 - p) * 80))
            if blocks < 2:
                blocks = 2
            small = cv2.resize(
                get_frame(t), (blocks, blocks), interpolation=cv2.INTER_LINEAR
            )
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return clip.transform(f)


class PixelateOut:
    def __init__(self, duration=1.5, min_blocks=4):
        self.d, self.min = duration, min_blocks

    def apply(self, clip):
        w, h = clip.size

        def f(get_frame, t):
            t0 = clip.duration - self.d
            if t < t0:
                return get_frame(t)
            p = (t - t0) / self.d
            blocks = int(w // (self.min + (1 - p) * 80))
            if blocks < 2:
                blocks = 2
            small = cv2.resize(
                get_frame(t), (blocks, blocks), interpolation=cv2.INTER_LINEAR
            )
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return clip.transform(f)


# ------------------------------------------------------------------
# 5.  CHROMATIC ABERRATION ZOOM
# ------------------------------------------------------------------
class ChromaticZoomIn:
    def __init__(self, duration=1.5, max_scale=1.05):
        self.d, self.s = duration, max_scale

    def apply(self, clip):
        w, h = clip.size

        def f(get_frame, t):
            if t >= self.d:
                return get_frame(t)
            p = t / self.d
            scale = 1 + p * (self.s - 1)
            frame = get_frame(t)
            b, g, r = cv2.split(frame)
            M = np.array(
                [[scale, 0, (w - w * scale) / 2], [0, scale, (h - h * scale) / 2]]
            )
            b = cv2.warpAffine(b, M, (w, h))
            r = cv2.warpAffine(r, M * np.array([[0.98, 0, 0], [0, 0.98, 0]]), (w, h))
            return cv2.merge((b, g, r))

        return clip.transform(f)


class ChromaticZoomOut:
    def __init__(self, duration=1.5, max_scale=1.05):
        self.d, self.s = duration, max_scale

    def apply(self, clip):
        w, h = clip.size

        def f(get_frame, t):
            t0 = clip.duration - self.d
            if t < t0:
                return get_frame(t)
            p = (t - t0) / self.d
            scale = 1 + (1 - p) * (self.s - 1)
            frame = get_frame(t)
            b, g, r = cv2.split(frame)
            M = np.array(
                [[scale, 0, (w - w * scale) / 2], [0, scale, (h - h * scale) / 2]]
            )
            b = cv2.warpAffine(b, M, (w, h))
            r = cv2.warpAffine(r, M * np.array([[0.98, 0, 0], [0, 0.98, 0]]), (w, h))
            return cv2.merge((b, g, r))

        return clip.transform(f)


if __name__ == "__main__":
    src = "/media/kornellewy/jan_dysk_3/auto_youtube/media/images/0a0a87b3-a553-495d-9413-0dac60d234b0.jpg"  # <-- your picture
    dur = 5  # base duration for the still
    out_dir = (
        "/media/kornellewy/jan_dysk_3/auto_youtube/helpers"  # where the demos land
    )

    clip = mp.ImageClip(src).with_duration(dur).with_fps(30)

    # 1.  DIFFUSION
    DiffusionIn(duration=2).apply(clip).write_videofile(
        f"{out_dir}/diffusion_in.mp4", preset="superfast"
    )
    DiffusionOut(duration=2).apply(clip).write_videofile(
        f"{out_dir}/diffusion_out.mp4", preset="superfast"
    )

    # 2.  GLITCH RGB
    GlitchRGBIn(duration=1).apply(clip).write_videofile(
        f"{out_dir}/glitchRGB_in.mp4", preset="superfast"
    )
    GlitchRGBOut(duration=1).apply(clip).write_videofile(
        f"{out_dir}/glitchRGB_out.mp4", preset="superfast"
    )

    # # 3.  RADIAL WIPE
    # RadialWipeIn(duration=1.5).apply(clip).write_videofile(
    #     f"{out_dir}/radialWipe_in.mp4", preset="superfast"
    # )
    # RadialWipeOut(duration=1.5).apply(clip).write_videofile(
    #     f"{out_dir}/radialWipe_out.mp4", preset="superfast"
    # )

    # 4.  PIXELATE
    PixelateIn(duration=1.5).apply(clip).write_videofile(
        f"{out_dir}/pixelate_in.mp4", preset="superfast"
    )
    PixelateOut(duration=1.5).apply(clip).write_videofile(
        f"{out_dir}/pixelate_out.mp4", preset="superfast"
    )

    # 5.  CHROMATIC ABERRATION ZOOM
    ChromaticZoomIn(duration=1.5).apply(clip).write_videofile(
        f"{out_dir}/chromaticZoom_in.mp4", preset="superfast"
    )
    ChromaticZoomOut(duration=1.5).apply(clip).write_videofile(
        f"{out_dir}/chromaticZoom_out.mp4", preset="superfast"
    )

    print("All 10 demo clips rendered ✔")
