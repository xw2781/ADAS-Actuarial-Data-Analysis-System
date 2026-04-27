from PIL import Image

png_path = r"E:\ADAS\library\icon\ADAS_icon_v2.png"
ico_path = r"E:\ADAS\library\icon\ADAS_icon_v2.ico"

img = Image.open(png_path)

sizes = [(16,16), (24,24), (32,32), (48,48), (64,64), (128,128), (256,256)]

img.save(
    ico_path,
    format="ICO",
    sizes=sizes
)

print("ICO created:", ico_path)