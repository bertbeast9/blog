


def read_last_line(filepath, lines_=0):
    """
    Reads the last line of a file efficiently.
    """
    with open(filepath, 'rb') as f:
        # Start at the end of the file
        f.seek(0, 2)
        # Get the current position (end of file)
        file_size = f.tell()
        ctr = 0
        # Iterate backwards byte by byte
        for i in range(2, file_size + 1):
            f.seek(-i, 2)  # Move cursor i bytes from the end
            char = f.read(1)
            if char == b'\n':
                if ctr == lines_:
                    # Found a newline, read the rest of the line
                    return f.readline().decode().strip()
                else:
                    ctr += 1
        
        # If no newline is found (e.g., single line file without trailing newline)
        f.seek(0)
        return f.read().decode().strip()