from __future__ import print_function

from PIL import Image, ImageChops
from PIL.GifImagePlugin import getheader, getdata


# --------------------------------------------------------------------
# sequence iterator

class ImageSequence:
    def __init__(self, im):
        self.im = im

    def __getitem__(self, ix):
        try:
            if ix:
                self.im.seek(ix)
            return self.im
        except EOFError:
            raise IndexError  # end of sequence


# --------------------------------------------------------------------
# straightforward delta encoding

def make_delta(fp, sequence):
    frames = 0

    previous = None

    for im in sequence:

        if not previous:

            # global header
            for s in getheader(im)[0] + getdata(im):
                fp.write(s)

        else:

            # delta frame
            delta = ImageChops.subtract_modulo(im, previous)

            bbox = delta.getbbox()

            if bbox:

                # compress difference
                for s in getdata(im.crop(bbox), offset=bbox[:2]):
                    fp.write(s)

            else:
                pass

        previous = im.copy()

        frames += 1

    fp.write(b';')

    return frames


# --------------------------------------------------------------------
# main hack

def compress(infile, outfile):
    # open input image, and force loading of first frame
    im = Image.open(infile)
    im.load()

    # open output file
    fp = open(outfile, "wb")

    seq = ImageSequence(im)

    # noinspection PyTypeChecker
    make_delta(fp, seq)

    fp.close()
