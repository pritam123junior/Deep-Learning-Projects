from PIL import Image
import pillow_heif
import exifread

def convert_heic_to_jpeg(heic_path, output_path):
    """Convert HEIC image to JPEG using Pillow and pillow-heif."""
    try:
        heif_image = pillow_heif.open_heif(heic_path)
        image = Image.frombytes(
            heif_image.mode, 
            heif_image.size, 
            heif_image.data
        )
        image.save(output_path, "JPEG")
        print(f"Converted {heic_path} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error converting HEIC to JPEG: {e}")
        return None

def extract_image_metadata(file_path):
    """Extract metadata from an image file."""
    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f)

        if not tags:
            print("No EXIF metadata found in the image.")
        else:
            print("\n=== EXIF Metadata ===")
            for tag, value in tags.items():
                print(f"{tag}: {value}")

            # Extract GPS coordinates if available
            gps_latitude = tags.get("GPS GPSLatitude")
            gps_longitude = tags.get("GPS GPSLongitude")
            gps_latitude_ref = tags.get("GPS GPSLatitudeRef")
            gps_longitude_ref = tags.get("GPS GPSLongitudeRef")

            if gps_latitude and gps_longitude:
                lat = convert_to_degrees(gps_latitude)
                lon = convert_to_degrees(gps_longitude)
                if gps_latitude_ref.values[0] != "N":
                    lat = -lat
                if gps_longitude_ref.values[0] != "E":
                    lon = -lon
                print("\nGPS Coordinates:")
                print(f"Latitude: {lat}, Longitude: {lon}")
                print(f"Google Maps Link: https://www.google.com/maps?q={lat},{lon}")
            else:
                print("\nNo GPS data found.")
    except Exception as e:
        print(f"Error extracting image metadata: {e}")

def convert_to_degrees(value):
    """Convert EXIF GPS coordinates to decimal degrees."""
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

if __name__ == "__main__":
    heic_file = "C:\iphone pic\IMG_1528.HEIC"  # Replace with the path to your HEIC file
    output_jpeg = "IMG_1528_converted.jpg"  # Path for the converted JPEG file

    # Convert HEIC to JPEG
    jpeg_path = convert_heic_to_jpeg(heic_file, output_jpeg)

    if jpeg_path:
        # Extract metadata from the converted JPEG file
        extract_image_metadata(jpeg_path)

    