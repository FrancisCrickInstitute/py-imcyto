import numpy as np
import math


def tilegen(img, tilesize = (512, 512), overlap = (32, 32), multi_class = False):
    '''Generator function to yield square tiles from an image with a defined overlap.'''

    if len(tilesize) > 2:
        tilesize = np.squeeze(tilesize)

    # setup:
    shape = img.shape
    offset = (tilesize[0] - overlap[0], tilesize[1] - overlap[1]) 
    r_tiles = int(math.ceil(shape[0]/(offset[1] * 1.0)))
    c_tiles = int(math.ceil(shape[1]/(offset[0] * 1.0)))

    # generate tiles:
    for i in range(r_tiles):
        for j in range(c_tiles):
            # create tile:
            crop = img[offset[1]*i:min(offset[1]*i+tilesize[1], shape[0]), offset[0]*j:min(offset[0]*j+tilesize[0], shape[1])]
            if crop.shape != tilesize:
                crop = np.pad(crop, ((0, tilesize[0] - crop.shape[0]), (0, tilesize[1] - crop.shape[1])), 'reflect')
            crop = crop / np.amax(crop)
            # crop = crop / 255
            crop = np.reshape(crop,crop.shape+(1,)) if (not multi_class) else crop
            crop = np.reshape(crop,(1,)+crop.shape)
            # crop = np.expand_dims(crop, axis=0)
            print(crop.shape)
            yield crop

def stitch_with_overlap(tiles, original_shape, tile_shape = (512, 512), overlap = (32, 32)):
    '''Stitch tiles back together with a defined overlap.'''

    # setup:
    offset = (tile_shape[0] - overlap[0], tile_shape[1] - overlap[1]) 
    r_tiles = int(math.ceil(original_shape[0]/(offset[1] * 1.0)))
    c_tiles = int(math.ceil(original_shape[1]/(offset[0] * 1.0)))

    stitched_shape = (r_tiles*tile_shape[1] - ((r_tiles-1)*overlap[1]), (c_tiles*tile_shape[1] - (c_tiles-1)*overlap[1]))
    stitched = np.zeros(stitched_shape)

    # stitch tiles:
    for i in range(r_tiles):
        for j in range(c_tiles):
            r_min = offset[1]*i
            r_max = min(offset[1]*i+tile_shape[1], stitched_shape[0])
            c_min = offset[0]*j
            c_max = min(offset[0]*j+tile_shape[0], stitched_shape[1])
            stitched[r_min:r_max, c_min:c_max] = tiles[i*c_tiles + j, :, :, 0]
    
    # crop to original size:
    stitched = stitched[:original_shape[0], :original_shape[1]]

    return stitched