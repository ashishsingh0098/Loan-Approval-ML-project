import lzma
import pickle

def compress_pickle_lzma(input_file_path, output_file_path):
  """
  Compresses a pickle file using lzma compression.

  Args:
      input_file_path (str): Path to the pickle file to compress.
      output_file_path (str): Path to save the compressed file.
  """
  
  
  with open('/home/ashish/VScode files/Python files/projects/ML project/Loan-Approval-ML-project/artifacts/model.pkl', "rb") as f_in:
      model = pickle.load(f_in)

  with lzma.open(output_file_path, "wb") as f_out:
      pickle.dump(model, f_out)



input_file = "model.pkl"
output_file = "model.pkl.lzma"
compress_pickle_lzma(input_file, output_file)
