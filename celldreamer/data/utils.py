indx = lambda a, i: a[i] if a is not None else None

class Args(dict):
    """
    Wrapper around a dictiornary to make its keys callable as attributes
    """
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        self.__dict__ = self
