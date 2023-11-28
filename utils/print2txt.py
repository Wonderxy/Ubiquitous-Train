import datetime
import os

def mkdir(path,mode="today"):
    strtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    strlist = strtime.split("-")
    if mode == "today":
        filePath = path+f"{strlist[1]}-{strlist[2]}"
    elif mode == "now":
        filePath = path+f"{strlist[3]}-{strlist[4]}-{strlist[5]}"
    else:
        raise ValueError("Mode ERROR")
    folder = os.path.exists(filePath)
    if not folder:
        os.makedirs(filePath,mode=0o777)           
        print("---  new folder  ---")
    else:
        print("---  already folder  ---")
    
    return filePath

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
    def __init__(self, file_path, file_name="default.txt", content="", mode='a',isAutoName=0):
        self.file_path = file_path
        self.file_name = file_name
        self.content = content
        self.mode = mode
        self.isAutoName = isAutoName
    
    def write_to_txt(self):
        """
        The write_to_txt method outputs the content to the corresponding txt file.
        """
        if self.isAutoName:
            new_path = mkdir(self.file_path,"today")
            strtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            strlist = strtime.split("-")
            new_path = new_path+f"/{strlist[3]}-{strlist[4]}-{strlist[5]}.txt"
        else:
            new_path = self.file_path+f"/{self.file_name}"
        with open(new_path, self.mode) as f:
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
    num_txt = PrintToTxt(f"C:/Users/14619/Desktop/print/","test.txt",[1,2,3,4,5],mode='a',isAutoName=1)
    num_txt.write_to_txt()
