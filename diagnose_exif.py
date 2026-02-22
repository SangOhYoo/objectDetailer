import os
import numpy as np
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper
from core.metadata import save_image_with_metadata
from unittest.mock import MagicMock

def create_complex_test_image(path):
    img = Image.new("RGB", (100, 100), color="white")
    
    # Define EXIF data
    zeroth_ifd = {
        piexif.ImageIFD.Make: u"Complex Camera",
        piexif.ImageIFD.Model: u"Complex Model",
        piexif.ImageIFD.Artist: u"Test Artist",
        piexif.ImageIFD.Software: u"Original Software",
    }
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: u"2023:01:01 00:00:00",
        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(u"Original Comment", encoding="unicode"),
    }
    gps_ifd = {
        piexif.GPSIFD.GPSLatitude: ((37, 1), (30, 1), (0, 1)),
    }
    
    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": {}, "thumbnail": None}
    exif_bytes = piexif.dump(exif_dict)
    
    # Add dummy XMP data
    info = PngImagePlugin.PngInfo()
    xmp_blob = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>Test XMP Data</x:xmpmeta>"
    info.add_text("XML:com.adobe.xmp", xmp_blob.decode('latin-1'))
    
    img.save(path, exif=exif_bytes, pnginfo=info)

def diagnose():
    test_input = "diag_input.png" # Using PNG to test PNG Info/XMP
    test_output = "diag_output.png"
    
    try:
        if os.path.exists(test_input): os.remove(test_input)
        create_complex_test_image(test_input)
        
        cv2_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        print(f"--- Diagnosing preservation ---")
        save_image_with_metadata(cv2_image, test_input, test_output)
        
        with Image.open(test_output) as out_img:
            raw_exif = out_img.info.get("exif")
            xmp_out = out_img.info.get("XML:com.adobe.xmp")

            if not raw_exif:
                print("[FAIL] No EXIF in output")
            else:
                exif_dict = piexif.load(raw_exif)
                soft = exif_dict['0th'].get(piexif.ImageIFD.Software)
                print(f"Preserved Make: {exif_dict['0th'].get(piexif.ImageIFD.Make)}")
                print(f"Preserved Artist: {exif_dict['0th'].get(piexif.ImageIFD.Artist)}")
                print(f"Preserved Software: {soft}")
                
                if exif_dict['0th'].get(piexif.ImageIFD.Artist) == b"Test Artist":
                    print("[SUCCESS] Artist preserved")
                else:
                    print(f"[FAIL] Artist mismatch or missing")

                if soft == b"Original Software":
                    print("[SUCCESS] Software tag preserved correctly (Not overwritten)")
                elif soft == b"ObjectDetailer_Ultimate":
                     print("[FAIL] Software tag was overwritten by our app!")
                else:
                     print(f"[NOTE] Software tag is: {soft}")

            if xmp_out:
                print(f"[SUCCESS] XMP preserved: {len(xmp_out)} bytes")
                if "Test XMP Data" in xmp_out:
                    print("[SUCCESS] XMP content verified")
            else:
                print("[FAIL] XMP missing in output")

    finally:
        if os.path.exists(test_input): os.remove(test_input)
        if os.path.exists(test_output): os.remove(test_output)


if __name__ == "__main__":
    diagnose()
