import moviepy as mp
import math
from PIL import Image
import numpy


def zoom_in_effect(clip, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t))),
        ]

        # The new dimensions must be even.
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        img = img.resize(new_size, Image.LANCZOS)

        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([x, y, new_size[0] - x, new_size[1] - y]).resize(
            base_size, Image.LANCZOS
        )

        result = numpy.array(img)
        img.close()

        return result

    return clip.transform(effect)


# size = (1920, 1080)

# images = [
#     "/media/kornellewy/jan_dysk_3/auto_youtube/media/images/alan_turing1.png",
#     "/media/kornellewy/jan_dysk_3/auto_youtube/media/images/comment_1652167749wc9xyXyXm17qI450G0tDG7,w400.jpg",
# ]

# slides = []
# for n, url in enumerate(images):
#     slides.append(mp.ImageClip(url).with_fps(25).with_duration(5).resized(size))

#     slides[n] = zoom_in_effect(slides[n], 0.04)


# video = mp.concatenate_videoclips(slides)
# video.write_videofile("zoomin.mp4")
