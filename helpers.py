import os

#Convert encoding of files in a folder
def convert_encoding(folder_path, from_encoding, to_encoding):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)        
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding = from_encoding) as file:
                file_contents = file.read()
            with open(file_path, 'w', encoding = to_encoding) as file:
                file.write(file_contents)
            print('Converted file:', filename)


#Print error message and file with line number
def error_handling(filename, target_line, error_message, print_line=False):
    filename = os.path.abspath(filename)
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            if target_line in line:
                print(error_message, '\n', filename+':'+str(line_number))
                if print_line:
                    print(target_line)
                return
    print(error_message, '\n', filename, target_line)


#display MacOS notification
display_notification = lambda notification: os.system("osascript -e 'display notification \"\" with title \""+notification+"\"'")