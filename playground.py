import configparser


config = configparser.ConfigParser()
config.read('example.ini')

for section in config.sections():
    print(dict(config.items(section)))