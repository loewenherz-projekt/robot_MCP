import subprocess

def get_usb_cam_info(dev):
    print(f"\n=== Kamera gefunden: {dev} ===")
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', dev, '--list-formats-ext'],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        if not output.strip():
            print("Keine Informationen gefunden (evtl. gesperrt oder kein Gerät).")
            return
        print(output)
    except Exception as e:
        print(f"Fehler bei {dev}: {e}")

def main():
    print("Starte Kamera-Erkennung für /dev/video0 bis /dev/video5...")

    for idx in range(6):  # 0 bis 5
        dev = f"/dev/video{idx}"
        try:
            with open(dev):
                get_usb_cam_info(dev)
        except Exception:
            # Gerät existiert nicht oder ist nicht zugreifbar
            continue

if __name__ == "__main__":
    main()
