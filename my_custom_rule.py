
from smdebug.rules.rule import Rule
import numpy as np

class MyCustomRule(Rule):
    def __init__(self, base_trial):
        super().__init__(base_trial)
        self.trial = base_trial
        self.counter = 0
        
        
    def invoke_at_step(self, step):
        
        #get predictions and labels
        predictions = np.argmax(self.trial.tensor('CrossEntropyLoss_input_0').value(step),axis=1)
        labels = self.trial.tensor('CrossEntropyLoss_input_1').value(step)

        #iterate over predictions and labels
        for prediction, label in zip(predictions, labels):
   
            #class is "Turn right ahead" and has been mistaken as "Turn left ahead" or vice versa:
            if prediction == 34 and label == 33 or prediction == 33 and label == 34:

                self.counter += 1
                
                #return True if the model has done this mistake more than 5 times.
                if self.counter > 5:
                    self.logger.info(f'Found {self.counter} where class 19 was mistaken as class 20 and vice versa')
                    return True
                
        return False
