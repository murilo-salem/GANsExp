import rasterio

v8 = "/home/lucas-fontoura/Documents/Pix2Pix/Data/Multispectral_satelital/Imagens/RRENIR_v8.tif"
v11 = "/home/lucas-fontoura/Documents/Pix2Pix/Data/Multispectral_satelital/Imagens/RRENIR_v11.tif"
v18 = "/home/lucas-fontoura/Documents/Pix2Pix/Data/Multispectral_satelital/Imagens/RRENIR_v18.tif"
r2 = "/home/lucas-fontoura/Documents/Pix2Pix/Data/Multispectral_satelital/Imagens/RRENIR_R2.tif"
r5 = "/home/lucas-fontoura/Documents/Pix2Pix/Data/Multispectral_satelital/Imagens/RRENIR_R5.tif"

with rasterio.open(v8) as a, rasterio.open(v11) as b, rasterio.open(v18) as c, rasterio.open(r2) as d, rasterio.open(r5) as e:
    
    print("Transforms:")
    print(a.transform)
    print(b.transform)
    print(c.transform)
    print(d.transform)
    print(e.transform)

    print("Bounds:")
    print(a.width, a.height)
    print(b.width, b.height)
    print(c.width, c.height)
    print(d.width, d.height)
    print(e.width, e.height)
    
    print("CRS: ")
    print(a.crs)
    print(b.crs)
    print(c.crs)
    print(d.crs)
    print(e.crs)
