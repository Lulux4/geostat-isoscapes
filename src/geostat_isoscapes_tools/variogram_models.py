import numpy as np
from abc import ABC,abstractmethod

LIST_MODELS = ['spherical','gaussian','exponential']

class VariogramModel(ABC):
    """ This class is the abstract class designating a variogram model"""
    @abstractmethod
    def get_model_func(self):
        pass

class GaussianModel(VariogramModel):
    """ This child-class of VariogramModel is a Gaussian model"""
    params = ['range','sill_ln','nugget_ln']
    def get_model_func(self):
        def f(h,range_,sill_ln,nugget_ln=-10):
            return np.exp(nugget_ln) + np.exp(sill_ln)*(1 - np.exp(-(h**2)/(range_**2)))
        return f
    
class SphericalModel(VariogramModel):
    """ This child-class of VariogramModel is a spherical model"""
    params = ['range','sill_ln','nugget_ln']
    def get_model_func(self):
        def f(h, range_,sill_ln, nugget_ln=-10):
            g = np.where(h <= range_,
                             np.exp(nugget_ln) + np.exp(sill_ln)*(1.5*(h/range_) -0.5*(h/range_)**3),
                             np.exp(nugget_ln) + np.exp(sill_ln))
            return g
        return f

class ExponentialModel(VariogramModel):
    params = ['range','sill','nugget_ln']
    """ This child-class of VariogramModel is an exp model"""
    def get_model_func(self):
        def f(h, range_, sill_ln, nugget_ln=-10):
            return np.exp(nugget_ln) + np.exp(sill_ln)*(1 - np.exp(-h/range_))
        return f

class CompositeModel(VariogramModel):
    """ This child-class of VariogramModel is a composite model that add two components (among
    these models : Gaussian, Exponential or Spherical).
    """
    def __init__(self, name:str):

        dict_model = {'spherical' : SphericalModel(),
                      'exponential': ExponentialModel(),
                      'gaussian': GaussianModel(),
                    }
        
        name1,name2 = name.split('+')
        try : 
            self.model1 = dict_model[name1]
            self.model2 = dict_model[name2]   
            self.params = ['nugget_ln','range1','sill1','range2','sill2']
        except Exception as e:
            raise e
            
    def get_model_func(self):
        f1 = self.model1.get_model_func()
        f2 = self.model2.get_model_func()
        
        def summed(h,nugget_ln,range1,sill_ln1,range2,sill_ln2):
            return f1(h,range_=range1,sill_ln=sill_ln1)+f2(h,range_=range2,sill_ln=sill_ln2)+np.exp(nugget_ln)

        return summed

def define_model(name:str):
    if name == 'spherical':
        return SphericalModel()
    elif name == 'gaussian':
        return GaussianModel()
    elif name== 'exponential':
        return ExponentialModel()
    else :
        try: 
            return CompositeModel(name)
        except Exception as e:
            print(e)
            raise ValueError(f"Cannot define a model of type {name}, only 'exponential','spherical' and 'gaussian' are supported.")

