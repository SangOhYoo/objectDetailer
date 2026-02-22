import os
import numpy as np
from PIL import Image
import piexif
import piexif.helper
from core.metadata import save_image_with_metadata
from unittest.mock import MagicMock

def create_test_image(path):
    # Create a 100x100 white image
    img = Image.new("RGB", (100, 100), color="white")
    
    # Define EXIF data
    # 0th IFD
    zeroth_ifd = {
        piexif.ImageIFD.Make: u"Test Camera",
        piexif.ImageIFD.Model: u"Test Model",
        piexif.ImageIFD.Copyright: u"Test User",
        piexif.ImageIFD.Software: u"Original Software"
    }
    # Exif IFD
    exif_ifd = {
        piexif.ExifIFD.DateTimeOriginal: u"2023:01:01 00:00:00",
        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(u"Original Comment", encoding="unicode")
    }
    
    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_bytes = piexif.dump(exif_dict)
    
    # Save with EXIF
    img.save(path, exif=exif_bytes)
    print(f"Created test image with EXIF: {path}")

def verify_exif():
    test_input = "test_input.jpg"
    test_output = "test_output.png"
    
    try:
        # 1. Create test image
        create_test_image(test_input)
        
        # 2. Mock config
        

        
        # 3. Dummy processed image (OpenCV BGR format)
        cv2_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2_image[:] = (0, 255, 0) # Green square
        
        # 4. Call preservation logic
        print("Running save_image_with_metadata...")
        print("Running save_image_with_metadata...")
        success = save_image_with_metadata(cv2_image, test_input, test_output)

        
        if not success:
            print("FAILED: save_image_with_metadata returned False")
            return

        # 5. Verify Output
        print("Verifying output EXIF...")
        out_img = Image.open(test_output)
        out_info = out_img.info
        
        raw_exif = out_info.get("exif")
        if not raw_exif:
            print("FAILED: Output image has no EXIF data")
            return
            
        exif_dict = piexif.load(raw_exif)
        
        # Check preserved tags
        make = exif_dict["0th"].get(piexif.ImageIFD.Make)
        model = exif_dict["0th"].get(piexif.ImageIFD.Model)
        copyright_tag = exif_dict["0th"].get(piexif.ImageIFD.Copyright)
        
        # piexif returns bytes for string tags, decode them
        def decode(val):
            if isinstance(val, bytes):
                return val.decode('utf-8').strip('\x00')
            return val

        print(f"Make: {decode(make)}")
        print(f"Model: {decode(model)}")
        print(f"Copyright: {decode(copyright_tag)}")
        
        assert decode(make) == "Test Camera", f"Make mismatch: {decode(make)}"
        assert decode(model) == "Test Model", f"Model mismatch: {decode(model)}"
        assert decode(copyright_tag) == "Test User", f"Copyright mismatch: {decode(copyright_tag)}"
        
        # Check Software (Should be preserved, NOT updated)
        software = exif_dict["0th"].get(piexif.ImageIFD.Software)
        print(f"Software: {decode(software)}")
        if decode(software) == "ObjectDetailer_Ultimate":
             print("[FAIL] Software tag was overwritten!")
             return False
        assert decode(software) == "Original Software", f"Software mismatch: {decode(software)}"

        
        user_comment_bytes = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
        if user_comment_bytes:
            try:
                user_comment = piexif.helper.UserComment.load(user_comment_bytes)
                print(f"UserComment: {user_comment}")
                
                # Should preserve original
                assert "Original Comment" in user_comment, "Original comment lost"
                # Should NOT have new prompt
                if "verified face" in user_comment:
                    print("[FAIL] UserComment contains new prompt data (should be removed)")
                    return False
            except Exception as e:
                print(f"Failed to load UserComment: {e}")
        
        # Check PNG Info (parameters) -> Should be EMPTY or just original if existed?
        # Our input didn't have PNG parameters, only EXIF.
        # So output parameters should be None or empty.
        params = out_info.get("parameters")
        print(f"PNG parameters: {repr(params)}")
        if params is not None and "verified face" in params:
             print("[FAIL] PNG parameters contain new prompt data!")
             return False
        
        print("\n[VERIFICATION] SUCCESS: All EXIF and Metadata checks passed!")
        return True


    except Exception as e:
        print(f"\n[VERIFICATION] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(test_input): os.remove(test_input)
        if os.path.exists(test_output): os.remove(test_output)

if __name__ == "__main__":
    if verify_exif():
        exit(0)
    else:
        exit(1)
