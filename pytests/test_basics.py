
import mergechannels

def test_default_function() -> None:
    print(dir(mergechannels))
    res = mergechannels.sum_as_string(10, 20)
    assert res == '30'

