import numpy as np
from typing import List

class KinematicTransform:
    '''
    Kinetmatic transform between frame A and B, or A->B.
    '''
    def __init__(self, t: np.ndarray, R: np.ndarray,
                 omega: np.ndarray, alpha: np.ndarray, v: np.ndarray, a: np.ndarray):
        '''
        :param t: The vector pointing from origin of A to B.
        :param R: The rotation matrix from A to B.
        :param omega: The angular velocity of B relative to A.
        :param alpha: The angular acceleration of B relative to A.
        :param v: The velocity of B relative to A.
        :param a: The acceleartion of B relative to A.
        '''
        self.t = t
        self.R = R
        self.omega = omega
        self.alpha = alpha
        self.v = v
        self.a = a

    @property
    def t(self) -> np.ndarray:
        '''Get the vector pointing from origin of A to B.'''
        return self._t

    @property
    def R(self) -> np.ndarray:
        '''Get the rotation matrix from A to B.'''
        return self._R

    @property
    def omega(self) -> np.ndarray:
        '''Get the angular velocity of B relative to A.'''
        return self._omega

    @property
    def alpha(self) -> np.ndarray:
        '''Get the angular acceleration of B relative to A.'''
        return self._alpha

    @property
    def v(self) -> np.ndarray:
        '''Get the velocity of B relative to A.'''
        return self._v

    @property
    def a(self) -> np.ndarray:
        '''Get the acceleration of B relative to A.'''
        return self._a

class KinematicChain:
    '''
    Represents a chain of kinematic transforms.
    '''
    def __init__(self, transforms: List[KinematicTransform]):
        '''
        :param transforms: A list of KinematicTransform objects.
        '''
        self._transforms = transforms

    @property
    def transforms(self) -> List[KinematicTransform]:
        '''Get the list of kinematic transforms.'''
        return self._transforms

    def add_transform(self, transform: KinematicTransform):
        '''Add a new kinematic transform to the chain.'''
        self._transforms.append(transform)

    def __getitem__(self, index: int) -> KinematicTransform:
        '''Get a kinematic transform by index.'''
        return self._transforms[index]

    def __len__(self) -> int:
        '''Get the number of kinematic transforms in the chain.'''
        return len(self._transforms)

    def __iter__(self):
        '''Iterate over the kinematic transforms.'''
        return iter(self._transforms)