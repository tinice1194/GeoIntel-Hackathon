from pathlib import Path
import rasterio                     

                      
PROJECT_ROOT = Path(r"G:\GIS_AI_PROJECT")

                                         
GEOTIFF_DIR = PROJECT_ROOT / "data" / "intermediate" / "geotiff"

                                   

                                   
                                                    
PIXEL_THRESHOLD = 6.0 * 1024**3                                                

                                                                    
MASK_RAM_THRESHOLD_GIB = 6.0             

def main():
    print(f"Scanning GeoTIFFs under: {GEOTIFF_DIR}\n")

    if not GEOTIFF_DIR.exists():
        print("GeoTIFF folder does not exist.")
        return

    any_found = False

    for tif in GEOTIFF_DIR.rglob("*.tif"):
                                                 
        if tif.name.upper().endswith("_MASK.TIF"):
            continue

        any_found = True
        try:
            with rasterio.open(tif) as src:
                width, height = src.width, src.height                        
        except Exception as e:
            print(f"{tif.name:70s}  ERROR opening file: {e}")
            continue

        pixels = width * height
                                      
        bytes_mask = pixels
        gib_mask = bytes_mask / (1024**3)

        flags = []
        if pixels > PIXEL_THRESHOLD:
            flags.append("MANY_PIXELS")
        if gib_mask > MASK_RAM_THRESHOLD_GIB:
            flags.append("MASK_RAM_HIGH")

        flag_str = ", ".join(flags) if flags else "OK"

        print(
            f"{tif.name:70s}  "
            f"{width:7d} x {height:<7d}  "
            f"{pixels/1e6:7.1f} Mpx  "
            f"mask≈{gib_mask:5.2f} GiB  "
            f"{flag_str}"
        )

    if not any_found:
        print("No .tif files found (excluding *_mask.tif).")


if __name__ == "__main__":
    main()
