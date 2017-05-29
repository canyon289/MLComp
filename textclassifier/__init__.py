import os 

def t_data():
    t_filepath = os.path.abspath(os.path.join(__file__, '..', 'data', 'test.txt'))
    f = open(t_filepath)
    print(f.read())
    return
