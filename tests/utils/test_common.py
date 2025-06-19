from feniax.utils import dict_difference, dict_merge, dict_deletebypath

def test_dict_difference_basic():
    d1 = {'a': 1, 'b': {'x': 10, 'y': 20}, 'c': 3}
    d2 = {'b': {'y': 20}}
    expected = {'a': 1, 'b': {'x': 10}, 'c': 3}
    assert dict_difference(d1, d2) == expected

def test_dict_difference_full_removal():
    d1 = {'a': 1, 'b': 2}
    d2 = {'a': 1, 'b': 2}
    expected = {}
    assert dict_difference(d1, d2) == expected

def test_dict_difference_partial_non_match():
    d1 = {'a': 1, 'b': {'x': 10, 'y': 20}}
    d2 = {'a': 2, 'b': {'x': 9}}
    expected = {'a': 1, 'b': {'x': 10, 'y': 20}}
    assert dict_difference(d1, d2) == expected

def test_dict_merge_basic():
    d1 = {'a': 1, 'b': {'x': 10, 'y': 20}}
    d2 = {'b': {'y': 99, 'z': 30}, 'c': 3}
    expected = {'a': 1, 'b': {'x': 10, 'y': 99, 'z': 30}, 'c': 3}
    assert dict_merge(d1, d2) == expected

def test_dict_merge_overwrite():
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 3, 'c': 4}
    expected = {'a': 1, 'b': 3, 'c': 4}
    assert dict_merge(d1, d2) == expected

def test_dict_merge_nested_conflict():
    d1 = {'a': {'nested': 1}}
    d2 = {'a': {'nested': {'deep': 2}}}
    expected = {'a': {'nested': {'deep': 2}}}
    assert dict_merge(d1, d2) == expected

def test_dict_deletebypath():
    d1 = {
        'a': 1,
        'b': {
            'x': 10,
            'y': 20
        }
    }
    
    dict_deletebypath(d1, 'b.x')
    assert {'a': 1, 'b': {'y': 20}} == d1
    
    dict_deletebypath(d1, 'a')
    assert  d1 == {'b': {'y': 20}}
    
