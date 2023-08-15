class PrintToTxt:
    """
    The constructor of the PrintToTxt class initializes object attributes.

    Parameters
    ----------
    param file_path : str, the storage location of the output txt file.
    param content : str or int or float or bool or list, the content to be written into the txt file.
    It can be a string, an integer, a float, a boolean value, or a list.
    param mode : str, the mode to write to the txt file. "w" is for overwriting (default), "a" is for appending.
    
    """
    def __init__(self, file_path, content, mode='a'):
        self.file_path = file_path
        self.content = content
        self.mode = mode
    
    def write_to_txt(self):
        """
        The write_to_txt method outputs the content to the corresponding txt file.
        """
        with open(self.file_path, self.mode) as f:
            if isinstance(self.content, str):
                f.write(self.content)
            elif isinstance(self.content, (int, float, bool)):
                f.write(str(self.content))
            elif isinstance(self.content, list):
                for item in self.content:
                    f.write(str(item))
                    f.write('\n')
            else:
                raise ValueError("Unsupported data type to write to txt.")


if __name__ == "__main__":
    txt = PrintToTxt("example.txt", "Hello, world!")
    txt.write_to_txt()

    num_txt = PrintToTxt("numbers.txt", [1, 2, 3, 4, 5], mode='a')
    num_txt.write_to_txt()
