import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from settings import *

def ndvi(nir: np.ndarray, red: np.ndarray)-> np.ndarray:
    top = nir - red
    bot = nir + red
    non_zero_mask = bot != 0
    valid_mask = ~np.isnan(top) & ~np.isnan(bot)
    final_mask = valid_mask & non_zero_mask
    result = np.divide(top, bot, where=final_mask, out=np.full_like(top, np.nan))
    return result

def ndre(nir: np.ndarray, re: np.ndarray)-> np.ndarray:
    top = nir-re
    bot = nir+re
    non_zero_mask = bot != 0
    valid_mask = ~np.isnan(top) & ~np.isnan(bot)
    final_mask = valid_mask & non_zero_mask
    result = np.divide(top, bot, where=final_mask, out=np.full_like(top, np.nan))
    return result

def sr(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    non_zero_mask = red != 0
    valid_mask = ~np.isnan(nir) & ~np.isnan(red)
    final_mask = valid_mask & non_zero_mask
    result = np.divide(nir, red, where=final_mask, out=np.full_like(nir, np.nan))
    return result

def srre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    non_zero_mask = red_edge != 0
    valid_mask = ~np.isnan(nir) & ~np.isnan(red_edge)
    final_mask = valid_mask & non_zero_mask
    result = np.divide(nir, red_edge, where=final_mask, out=np.full_like(nir, np.nan))
    return result

def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, g =2.5, c1 = 6, c2 = 7.5, l = 1, scale = 0.0001) -> np.ndarray:
    #for some reason you have to scale it down? See below:
    #https://community.esri.com/t5/arcgis-spatial-analyst-questions/formula-for-enhanced-vegetation-index-evi-in/td-p/32453
    top = (nir*scale-red*scale) *g
    bot = nir*scale+(c1*red*scale)-(c2*blue*scale)+l 
    non_zero_mask = bot != 0
    valid_mask = ~np.isnan(top) & ~np.isnan(bot)
    final_mask = valid_mask & non_zero_mask
    result = np.divide(top, bot, where=final_mask, out=np.full_like(top, np.nan))
    return result

def msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir2_1 = 2*nir + 1
    nir_red = nir-red
    sqr_root = np.sqrt((nir2_1)**2 - 8*nir_red)
    top = nir2_1 - sqr_root
    final_mask = ~np.isnan(top)
    result = np.divide(top, 2, where=final_mask, out=np.full_like(top, np.nan))
    return result


def scaleMinMax(x : np.ndarray) -> np.ndarray:
    return((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)))

def get_bands(min : int ,max: int(),image):
    band = 0
    for band in range(min,max+1):
        band += np.array(image.GetRasterBand(band).ReadAsArray())
    return scaleMinMax(band)

def make_and_save_graph(title, data):
    plt.figure()
    plt.title(title)
    plt.imshow(data,cmap = "RdYlGn")
    plt.colorbar(label='Intensity')
    plt.savefig(title +'.png')
    plt.close()

def heatmaps(name, to_return = False):
    indices = {"NDVI": None, "NDRE": None,  "SR":None,  "SRRE":None, "EVI" : None, "MSAVI" : None}
    ds = gdal.Open(name)
    if ds == None:
        return indices

    #get the band values
    if type(GREEN_BAND) == type(1):
        green = scaleMinMax(np.array(ds.GetRasterBand(GREEN_BAND).ReadAsArray()))
    else:
        green = get_bands(GREEN_BAND[0],GREEN_BAND[-1],ds)

    if type(BLUE_BAND) == type(1):
        blue = scaleMinMax(np.array(ds.GetRasterBand(BLUE_BAND).ReadAsArray()))
    else:
        blue = get_bands(BLUE_BAND[0],BLUE_BAND[-1],ds)
    
    if type(RED_BAND) == type(1):
        red = scaleMinMax(np.array(ds.GetRasterBand(RED_BAND).ReadAsArray()))
    else:
        red = get_bands(RED_BAND[0],RED_BAND[-1],ds)
    
    if type(RED_EDGE_BAND) == type(1):
        red_edge = scaleMinMax(np.array(ds.GetRasterBand(RED_EDGE_BAND).ReadAsArray()))
    else:
        red_edge = get_bands(RED_EDGE_BAND[0],RED_EDGE_BAND[-1],ds)

    if type(NIR_BAND) == type(1):
        nir = scaleMinMax(np.array(ds.GetRasterBand(NIR_BAND).ReadAsArray()))
    else:
        nir = get_bands(NIR_BAND[0],NIR_BAND[-1],ds)

    indices["NDVI"] = ndvi(nir,red)
    indices["NDRE"] = ndre(nir, red_edge)
    indices["SR"] = sr(nir, red)
    indices["SRRE"] = srre(nir, red_edge)
    indices["EVI"] = evi(nir, red, blue)
    indices["MSAVI"] = msavi(nir, red)
    if to_return:
        return indices
    for key in indices:
        if isinstance(indices[key], np.ndarray):
            make_and_save_graph(key,indices[key])


def heatmap_selector(name, selector, to_return = False):
    ds = gdal.Open(name)

    """Press 1 for NDVI heatmap
    Press 2 for NDRE heatmap,
    Press 3 for SR heatmap,
    Press 4 for SRRE heatmap """

    if selector == 1: #NDVI from nir and red
        if type(RED_BAND) == type(1):
            red = scaleMinMax(np.array(ds.GetRasterBand(RED_BAND).ReadAsArray()))   
        else:
            red = get_bands(RED_BAND[0],RED_BAND[-1],ds)

        if type(NIR_BAND) == type(1):
            nir = scaleMinMax(np.array(ds.GetRasterBand(NIR_BAND).ReadAsArray()))
        else:
            nir = get_bands(NIR_BAND[0],NIR_BAND[-1],ds)
        heatmap = ndvi(nir,red)

    elif selector == 2: #NDRE from nir and re
        if type(RED_EDGE_BAND) == type(1):
            red_edge = scaleMinMax(np.array(ds.GetRasterBand(RED_EDGE_BAND).ReadAsArray()))   
        else:
            red_edge = get_bands(RED_EDGE_BAND[0],RED_EDGE_BAND[-1],ds)

        if type(NIR_BAND) == type(1):
            nir = scaleMinMax(np.array(ds.GetRasterBand(NIR_BAND).ReadAsArray()))
        else:
            nir = get_bands(NIR_BAND[0],NIR_BAND[-1],ds)
        heatmap = ndre(nir, red_edge)
    
    elif selector == 3: # SR from nir and red
        if type(RED_BAND) == type(1):
            red = scaleMinMax(np.array(ds.GetRasterBand(RED_BAND).ReadAsArray()))   
        else:
            red = get_bands(RED_BAND[0],RED_BAND[-1],ds)

        if type(NIR_BAND) == type(1):
            nir = scaleMinMax(np.array(ds.GetRasterBand(NIR_BAND).ReadAsArray()))
        else:
            nir = get_bands(NIR_BAND[0],NIR_BAND[-1],ds)
        heatmap = sr(nir, red)

    elif selector == 4: # SRRE from nir and re
        if type(RED_EDGE_BAND) == type(1):
            red_edge = scaleMinMax(np.array(ds.GetRasterBand(RED_EDGE_BAND).ReadAsArray()))   
        else:
            red_edge = get_bands(RED_EDGE_BAND[0],RED_EDGE_BAND[-1],ds)

        if type(NIR_BAND) == type(1):
            nir = scaleMinMax(np.array(ds.GetRasterBand(NIR_BAND).ReadAsArray()))
        else:
            nir = get_bands(NIR_BAND[0],NIR_BAND[-1],ds)
        heatmap = srre(nir, red_edge)
    
    if to_return:
        return heatmap
    
    else:
        plt.figure()
        plt.imshow(heatmap)
        plt.savefig()
        plt.show()




if __name__ == '__main__':
    heatmaps("example.tif")
    arrays = heatmaps("example.tif", to_return=True) #returns the indices as a dictionary of np.arrays. Graphs wont be saved to avoid cluttering
    
