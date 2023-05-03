
import re

TRANSCRIPT_ENCODING='iso-8859-1'

#parse file
def parser(filename):
    with open(filename, 'r', encoding=TRANSCRIPT_ENCODING) as f:
        text = f.read()

        lines = text.split('\n')
        interview = {}
        section = ''
        interview['Sections'] = []
        for line in lines:
            #Initial key-value pairs
            if line.startswith('#') and  re.search(r':[ \t]', line):
                key, value = line.split(':', 1)
                interview[key[1:]] = value.strip()
            
            #Section headers
            elif line.startswith('#'):
                if section != '':
                    interview[section] = interview[section].strip()
                section = re.sub(r'[\d:]+', '', line[1:]).strip()
                if section in ['START', 'END']:
                    section = ''
                else:
                    interview[section] = ''
                    if section != 'FIELD NOTES':
                        interview['Sections'].append(section)
            #Section content
            elif not line.startswith('#') and section != '':
                interview[section] += line + '\n'
        return interview


