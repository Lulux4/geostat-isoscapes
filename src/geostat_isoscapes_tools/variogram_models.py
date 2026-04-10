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
    params = ['range','sill']
    def get_model_func(self):
        def f(h,range_,sill):
            return sill*(1 - np.exp(-(h**2)/(range_**2)))
        return f
    
class SphericalModel(VariogramModel):
    """ This child-class of VariogramModel is a spherical model"""
    params = ['range','sill']
    def get_model_func(self):
        def f(h, range_,sill):
            g = np.where(
                h <= range_,
                sill*(1.5*(h/range_) -0.5*(h/range_)**3),
                sill)
            return g
        return f

class ExponentialModel(VariogramModel):
    params = ['range','sill']
    """ This child-class of VariogramModel is an exp model"""
    def get_model_func(self):
        def f(h, range_, sill):
            return sill*(1 - np.exp(-h/range_))
        return f

class Nugget(VariogramModel):
    params = ['nugget']
    """ This child-class of VariogramModel is a nugget effect model"""
    def get_model_func(self):
        def f(h, nugget):
            return nugget*np.ones_like(h)
        return f

class CompositeModel(VariogramModel):
    """ This child-class of VariogramModel is a composite model that add two components (among
    these models : Gaussian, Exponential, Spherical or Nugget).
    """
    def __init__(self, name:str):

        dict_model = {'spherical' : SphericalModel(),
                      'exponential': ExponentialModel(),
                      'gaussian': GaussianModel(),
                      'nugget':Nugget(),
                    }

        try : 
            models_list = name.split('+')

            for model in models_list :
                if model not in dict_model.keys():
                    raise ValueError(f"Cannot define a model of type {model}, only 'exponential','spherical' and 'gaussian', or a composition of two of them (e.g 'gaussian+spherical')are supported.")
            self.models = [(model_name,dict_model[model_name]) for model_name in models_list]
            self.params_with_model_name = [(model_name,params) for model_name,model in self.models for params in model.params]

            # if some models share parameters with the same name, we need to rename them to avoid confusion.
            # For example, the list of parameters [('spherical',['range','sill']),('gaussian',['range','sill']),('nugget',['nugget'])] would become ['range_spherical','sill_spherical','range_gaussian','sill_gaussian','nugget']
            list_params = [ps for _,ps in self.params_with_model_name]
            if list_params.count('range') > 1:
                for i,(m,p) in enumerate(self.params_with_model_name): # name of the model and params list
                    if 'range' in p :
                        # then modify the parameter name to include the model name, e.g 'range_spherical' instead of 'range'
                        self.params_with_model_name[i] = (m,[param if param != 'range' else f'range_{m}' for param in p]) 
            if list_params.count('sill') > 1:
                for i,(m,p) in enumerate(self.params_with_model_name): # name of the model and params list
                    if 'sill' in p :
                        # then modify the parameter name to include the model name, e.g 'sill_spherical' instead of 'sill'
                        self.params_with_model_name[i] = (m,[param if param != 'sill' else f'sill_{m}' for param in p])
            
            self.params = [params for _,params in self.params_with_model_name]

        except Exception as e:
            raise e
            
    def get_model_func(self):
        funcs = [(model_name, model.get_model_func()) for model_name, model in self.models]
        
        def summed(h,*args):
            ''' args should be ordered as in [range,sill,range,sill,...,nugget], following the order the model_names except for the nugget if there is one.'''
            if len(args) != len(self.params):
                raise ValueError(f"The number of parameters provided ({len(args)}) does not match the number of parameters expected by the composite model ({len(self.params)}).")
            sum = 0
            j=0
            for model_name, func in funcs:
                if model_name =='nugget':
                    sum += func(h,*[args[-1]]) # nugget is always the last one
                else :
                    n_params = 1 if model_name == 'nugget' else 2 # nugget has only one parameter, the others have two parameters (range and sill)
                    params_values = [args[j+i] for i in range(n_params)] # it is always 'range_***' followed by 'sill_***', normally
                    j=j+n_params
                    sum += func(h,*params_values)
            return sum
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
            raise ValueError(f"Cannot define a model of type {name}, only 'exponential','spherical' and 'gaussian', or a composition of two of them (e.g 'gaussian+spherical')are supported.")

