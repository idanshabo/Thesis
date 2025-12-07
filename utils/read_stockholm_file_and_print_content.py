def read_stockholm_file_and_print_content(stockholm_file_path):
    try:
        with open(stockholm_file_path, 'r') as f:
            print("File contents:")
            # Initialize variables
            sequences = {}

            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Skip annotation lines and format identifier
                if line.startswith('#') or line == "//":
                    continue

                # Process sequence lines
                if line:
                    # Stockholm format has sequence ID and sequence separated by whitespace
                    try:
                        seq_id, sequence = line.split()
                        if seq_id in sequences:
                            sequences[seq_id] += sequence
                        else:
                            sequences[seq_id] = sequence
                    except ValueError:
                        continue

            # Print the sequences
            print(f"\nFound {len(sequences)} sequences:")
            for seq_id, sequence in sequences.items():
                print(f"\nID: {seq_id}")
                print(f"Sequence: {sequence[:604]}...")  # Print first 60 characters
                print(f"Length: {len(sequence)}")

    except FileNotFoundError:
        print("Error: File not found")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
