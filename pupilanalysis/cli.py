

def simple_select(list_of_strings):
    '''
    Simple command line user interface for selecting a string
    from a list of many strings.
    
    Returns the string selected.
    '''

    for i_option, option in enumerate(list_of_strings):
        print('{}) {}'.format(i_option+1, option))

    while True:
        sel = input('Type in selection by number: ')

        try:
            sel = int(sel)
        except TypeError:
            print('Please input a number')
            continue

        if 1 <= sel <= len(list_of_strings):
            return list_of_strings[sel-1]


