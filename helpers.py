import os

#display MacOS notification
display_notification = lambda notification: os.system("osascript -e 'display notification \"\" with title \""+notification+"\"'") 