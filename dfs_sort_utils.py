#!/usr/bin/env python
'''
Functions from this script are utilised to sort an array of tree indices in
dfs order
'''
def dfs_less_sign(str1, str2):
    '''
    This function is used to sort list of tree indices in dfs order.
    :return: True if str1 < str2 in some sense of author
    :rtype: bool
    '''

    # extracting the index
    idx_l_1 = str1.split('. ')[0].split('.')
    idx_l_2 = str2.split('. ')[0].split('.')

    # if the second string is empty
    if idx_l_2[0] == '' and len(idx_l_2) == 1:
        return False
    # if the fisrt string is empty and the second is not
    elif idx_l_1[0] == '' and len(idx_l_1) == 1:
        return True

    i = 0  # max common start string length
    min_len = min(len(idx_l_1), len(idx_l_2))
    while i < min_len and idx_l_1[i] == idx_l_2[i]:
        i += 1


    if i < min_len:
        n1 = int(idx_l_1[i])
        n2 = int(idx_l_2[i])
        return n1 < n2
    else: # if i == min_len
        # if the second_string == start(first_string)
        if i == len(idx_l_2):
            return False
        # if the first string == start(first_string)
        return True

def merge_sort(array, less_sign_func):
    """
    Sorting algorithm with time complexity O(n*logn)
    :param array: list of elements to sorte
    :param less_sing_func: function that defines less sign operator for elements
        of arry
    :type: func: (obj1, obj2) --> bool
    :return: sorted array
    :rtype: list
    """
    if len(array) == 1:
        return array
    m = len(array)//2
    array_a = merge_sort(array[:m], less_sign_func)
    array_b = merge_sort(array[m:], less_sign_func)
    array_sorted = merge(array_a, array_b, less_sign_func)
    return array_sorted

def merge(array_1, array_2, less_sign_func):
    """
    Merges two sorted arrays in to one sorted array
    Time Complexity O(len(array1) + len(array2))
    """
    result = []
    n1 = len(array_1)
    n2 = len(array_2)
    i = j = 0
    while (i < n1) and (j < n2):
        if less_sign_func(array_1[i], array_2[j]):
        # if array_1[i] < array_2[j]:
            result.append(array_1[i])
            i += 1
        else:
            result.append(array_2[j])
            j += 1

    while i < n1:
        result.append(array_1[i])
        i += 1

    while j < n2:
        result.append(array_2[j])
        j += 1

    return result

def dfs_sort(array):
    return merge_sort(array, dfs_less_sign)

def print_dfs_order_mistakes(indices_list):

    for i in range(len(indices_list)-1):

        # separating node.index from string f'{node.index}. {node.name}'
        curr_str = indices_list[i].split('. ')[0]
        next_str = indices_list[i+1].split('. ')[0]

        curr = list(map(int, curr_str.split('.')))
        next_ = list(map(int, next_str.split('.')))

        j = 0  # max common start substring length
        min_len = min(len(curr), len(next_))
        while j < min_len and curr[j] == next_[j]:
            j += 1

        message = f'{indices_list[i]} --> {indices_list[i+1]}'
        is_error = False

        if j < min_len:  # if in both strings indices are left
            # if not (the remainder of next is just one index and that index
                # is the first remainder of curr + 1)
            if not ((len(next_) - j) == 1 and next_[j] == curr[j] + 1):
                is_error = True
        elif j == min_len: # if one is start substring of another one
            # 3.6.36.7.8.1.2  -> 3.6.36.7.8.1.2.1 is the only good situation
            if not (len(curr) + 1 == len(next_) and next_[j] == 1):
                is_error = True
        else:
            print('Go debug. You never should get here!')

        if is_error:
            print(message)

if __name__ == '__main__':
    # testing

    import json
    import os
    user_folder = os.path.expanduser('~')
    root_folder = os.path.join(user_folder, 'Desktop/lawAI/data')
    file_path = os.path.join(root_folder, 'dispute_categories.json')
    with open(file_path, 'r') as f:
        disp_cat1 = json.load(f)

    with open(file_path, 'r') as f:
        disp_cat2 = json.load(f)

    # print(disp_cat)
    import numpy as np
    n_trials = 1000
    for _ in range(n_trials):
        np.random.shuffle(disp_cat1)
        bool1 = disp_cat1 == disp_cat2
        bool2 = dfs_sort(disp_cat1) == disp_cat2
        if not bool1 and bool2:
            pass
        else:
            print('Warning')
            print(bool1, bool2)
    print('OK')
