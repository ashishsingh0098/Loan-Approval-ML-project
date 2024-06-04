import lzma
import pickle


def decompress_pickle_lzma(compressed_file_path, output_file_path):
    """
    Decompresses a pickle file compressed with lzma.

    Args:
        compressed_file_path (str): Path to the compressed pickle file.
        output_file_path (str): Path to save the decompressed model.
    """
    with lzma.open(compressed_file_path, "rb") as f_in:
        model = pickle.load(f_in)

    with open(output_file_path, "wb") as f_out:
        pickle.dump(model, f_out)


compressed_file = "your_model.pkl.lzma"  # Replace with your compressed file path
decompressed_file = "decompressed_model.pkl"  # Replace with desired output path

decompress_pickle_lzma(compressed_file, decompressed_file)  # Call the function (if defined)
