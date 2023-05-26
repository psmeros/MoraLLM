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

#display MacOS notification
display_notification = lambda notification: os.system("osascript -e 'display notification \"\" with title \""+notification+"\"'")