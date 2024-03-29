import os

def get_label():
    with open('label.txt', 'a') as f:
        with open('skes_available_name.txt', 'r') as s:
            file_name_list = s.readlines()
            for file_name in file_name_list:
                label = int(file_name[file_name.find('A')+1:file_name.find('A')+4])
                f.write(str(label)+'\n')
    f.close()

def get_performer():
    with open('performer.txt', 'a') as f:
        with open('skes_available_name.txt', 'r') as s:
            file_name_list = s.readlines()
            for file_name in file_name_list:
                performer = int(file_name[file_name.find('P')+1:file_name.find('P')+4])
                f.write(str(performer)+'\n')
    f.close()

if __name__ == '__main__':
    get_label()
    get_performer()