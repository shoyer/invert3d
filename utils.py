import numpy as np


class MetaArray(np.ndarray):
    """
    Subclass of numpy.ndarray which stores metadata supplied as named arguments
    in a dictionary under the `metadata` attribute.

    Metadata keys can also be accessed directly as arrays attributes. Hence they
    are checked upon creating a MetaArray instance against pre-existing
    attributes to ensure there are no collisions.

    Example Usage
    -------------
    >>> import numpy as np
    >>> x = MetaArray(np.arange(5), extra='saved')
    >>> x.extra
    'saved'

    References
    ----------
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """

    def __new__(cls, input_array, **metadata):
        """Create a new MetaArray object"""
        obj = np.asarray(input_array).view(cls)
        obj.metadata = metadata
        obj.check_metadata()
        return obj

    def __array_finalize__(self, obj):
        """Ensure a modified MetaArray is still a MetaArray"""
        if obj is not None:
            self.metadata = getattr(obj, 'metadata', {})

    def check_metadata(self):
        """Check to make sure metadata keys are not already attributes"""
        for key in self.metadata.iterkeys():
            if key in dir(self):
                raise AttributeError(
                    '{0} is already a named attribute'.format(key))

    def __getattr__(self, name):
        """Attribute style acccess for metadata"""
        try:
            return self.metadata[name]
        except KeyError:
            raise AttributeError('object has no attribute {0}'.format(name))

    def __repr__(self):
        metadata_str = ''.join(', {0}={1}'.format(k, v)
                               for k, v in self.metadata.iteritems())
        return '{0}({1}{2})'.format(type(self).__name__,
                                    np.ndarray.__repr__(self),
                                    metadata_str)

    def __reduce__(self):
        """Used by pickle"""
        state = np.ndarray.__reduce__(self)
        state[2] = (state[2], self.metadata)
        return tuple(state)

    def __setstate__(self, (array_state, metadata)):
        """Used by pickle"""
        np.ndarray.__setstate__(self, array_state)
        self.metadata = metadata
