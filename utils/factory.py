from methods.learner import AVLearner

def get_model(model_name, args, logfilename):
    name = model_name.lower()
    options = {'avlearner': AVLearner,
               }
    return options[name](args, logfilename)
